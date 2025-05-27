#include <Arduino.h>
#include <driver/i2s.h>
#include <freertos/queue.h>
#include <esp_timer.h>
#include <esp_attr.h>
#include <dsps_math.h>
#include <dsps_fft2r.h>
#include <dsps_wind.h>
#include <cmath>

// Display
#include <Wire.h>  
#include "HT_SSD1306Wire.h"

// LoRaWAN
#include "LoRaWan_APP.h"

static SSD1306Wire display(0x3c, 500000, SDA_OLED, SCL_OLED, GEOMETRY_128_64, RST_OLED);

// I2S + DSP + TFLite model come qui…

#define NUM_SCORES 5
float scores[NUM_SCORES];          // punteggi prodotti da InferenceTask

// -----------------------------------------------------------
// OTAA params (oppure ABP) – come nel tuo template
uint8_t devEui[] = { /* … */ };
uint8_t appEui[] = { /* … */ };
uint8_t appKey[] = { /* … */ };
// …

// Application data (max payload size definito in commissioning.h)
static uint8_t appData[LORAWAN_APP_DATA_MAX_SIZE];
static uint8_t appDataSize;

// -----------------------------------------------------------
/**
 * Prepara il payload copiando i 5 punteggi quantizzati in 5 byte.
 */
static void prepareTxFrame(uint8_t port) {
  // quantizzo i float [0.0–1.0] in 0–100 (o usa il tuo range)
  appDataSize = NUM_SCORES;
  for (int i = 0; i < NUM_SCORES; i++) {
    uint8_t q = (uint8_t)constrain(scores[i] * 100.0f, 0.0f, 100.0f);
    appData[i] = q;
  }
  // imposta il port (se ti serve)
  appPort = port;
}

// -----------------------------------------------------------
void setup() {
  Serial.begin(115200);
  // … inizializza I2S, display, TFLite, ecc. …

  // LoRaWAN init
  LoRaWAN.init(loraWanClass, loraWanRegion);
#if(LORAWAN_DEVEUI_AUTO)
  LoRaWAN.generateDeveuiByChipID();
#endif
  LoRaWAN.setDefaultDR(3);
}

void loop() {
  switch (deviceState) {
    case DEVICE_STATE_INIT:
      LoRaWAN.join();
      deviceState = DEVICE_STATE_JOIN;
      break;

    case DEVICE_STATE_JOIN:
      // aspetta join completato…
      break;

    case DEVICE_STATE_SEND:
      // qui i punteggi sono già stati scritti da InferenceTask
      prepareTxFrame(appPort);
      LoRaWAN.send();            // invia appData/appDataSize
      deviceState = DEVICE_STATE_CYCLE;
      break;

    case DEVICE_STATE_CYCLE:
      {
        uint32_t delay_ms = appTxDutyCycle + randr(-APP_TX_DUTYCYCLE_RND, APP_TX_DUTYCYCLE_RND);
        LoRaWAN.cycle(delay_ms);
        deviceState = DEVICE_STATE_SLEEP;
      }
      break;

    case DEVICE_STATE_SLEEP:
      LoRaWAN.sleep(loraWanClass);
      break;

    default:
      deviceState = DEVICE_STATE_INIT;
      break;
  }
}

// -----------------------------------------------------------
// All’interno di InferenceTask, subito dopo aver scritto
// il vettore `scores[]`, mettere  deviceState = DEVICE_STATE_SEND;
// oppure notificare il loop con un flag globale/thread-safe.
