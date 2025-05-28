
#include <heltec.h>
#include <lmic.h>
#include <hal/hal.h>

// … your I²S, DSPS, TFLite includes and setup here …

// LoRaWAN OTAA keys (LSB order)
static const u1_t PROGMEM DEVEUI[8] = {0x70, 0xB3, 0xD5, 0x7E, 0xD0, 0x06, 0xFF, 0x3A};
static const u1_t PROGMEM APPEUI[8] = {0x70, 0xB3, 0xD5, 0x7E, 0xD0, 0x04, 0xB0, 0xA1};
static const u1_t PROGMEM APPKEY[16] = {0x2C, 0xCF, 0xB0, 0x32, 0x3F, 0x7C, 0xAB, 0xB7, 0xD2, 0x80, 0x1D, 0x43, 0x08, 0xEF, 0x03, 0x77};


const lmic_pinmap lmic_pins = {
  .nss   = 18, .rxtx = LMIC_UNUSED_PIN, .rst  = 14,
  .dio   = {26, 33, 32},
};

// forward-declare your inference callback
void send_scores_via_lora(const float* scores, int n);


// Packs up to 5 floats 
// and enqueues a LoRaWAN uplink on port 1.

void send_scores_via_lora(const float* scores, int n) {
    // Simple quantization: map 0.0–1.0 score to 0–255
    uint8_t payload[5];
    for (int i = 0; i < n; i++) {
        float v = scores[i];
        if (v < 0.f) v = 0.f;
        if (v > 1.f) v = 1.f;
        payload[i] = (uint8_t)roundf(v * 255.0f);
    }

    // Wait until no TX pending
    if (LMIC.opmode & OP_TXRXPEND) {
        Serial.println(F("TX pending, skipping"));
        return;
    }

    // Port 1, unconfirmed uplink
    LMIC_setTxData2(1, payload, n, 0);
    Serial.printf("LoRa: queued %d scores\n", n);
}

// ----------------------------------------------------------------------------
// In setup() add your LMIC initialization:

void setupLora() {

    Heltec.begin(true, true, true);

    //LoRaWAN init:
    os_init();
    LMIC_reset();
    LMIC_setLinkCheckMode(0);
    LMIC_startJoining();
}
