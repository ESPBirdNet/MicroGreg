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

static SSD1306Wire  display(0x3c, 500000, SDA_OLED, SCL_OLED, GEOMETRY_128_64, RST_OLED); // addr , freq , i2c group , resolution , rst

// TFLite
#include "TFLiteModel.h"

TFLiteModel tfliteModel;

//============================== Pinout ==============================
#define MIC_SD   7    // data out from INMP441
#define MIC_WS   5    // word select (LRCLK)
#define MIC_SCK  6    // bit clock (BCLK)

// I2C config to display spectrogram from pi pico
#define I2C_SDA  3
#define I2C_SCL  4
#define SLAVE_ADDR 0x42

//============================== Configuration ==============================
#define I2S_NUM                     I2S_NUM_0
#define SAMPLE_BITS                 I2S_BITS_PER_SAMPLE_32BIT
#define SAMPLE_RATE_HZ              32000

// DMA buffer count & length (1024 samples each)
#define DMA_BUF_LEN                 1024                    // 16ms of audio
#define DMA_BUF_COUNT               32 + 16                 // 1s of stored audio + next quarter of second + residual
#define DMA_BUF_COUNT_INT32         2

// FFT / spectrogram parameters
#define WIN_BUFFERS            32                           // around 1 second of audio
#define HOP_BUFFERS            16                           // After collecting how many new buffers should a new spectrogram be computed? Once every 1/HOP_BUFFERS seconds!
#define FFT_SIZE               1024                         // 2 * DMA_BUF_LEN for simple hop in FFT computation and good enough frequency resolution
#define NUM_BANDS              128

#define SPECTROGRAM_WIDTH      (DMA_BUF_LEN * WIN_BUFFERS) / FFT_SIZE                          // 32 and no hop
// #define SPECTROGRAM_WIDTH      (DMA_BUF_LEN * WIN_BUFFERS - FFT_SIZE) / DMA_BUF_LEN - 1     // 61 (not 63!) - 50% Hop

//============================== Globals ==============================
#define SPECTRUM_FLOOR          30
#define SPECTRUM_CEIL           158
#define SPECTRUM_SCALE          (256 / (SPECTRUM_CEIL - SPECTRUM_FLOOR))

//============================== Globals ==============================
// DMA buffers
static int16_t * audioBufs[DMA_BUF_COUNT];
static volatile size_t bufIndex = 0;

static int32_t pingPongBuf[DMA_BUF_COUNT_INT32][DMA_BUF_LEN];
static volatile int currentBuf = 0;

// I2S event queue
static QueueHandle_t i2s_evt_queue  = nullptr;

// Count buffers to trigger spectrogram
static volatile int bufferCount = 0;

// Hann window & FFT arrays
static float * hann_window;
static int16_t * fft_buffer;
static int32_t * power_spectrum;
static float   band_energy;
// static float   band_energies[NUM_BANDS];

static uint8_t * spectrogram_data = nullptr;

// Task handles
static TaskHandle_t SpectrogramTaskHandle = nullptr;
static TaskHandle_t InferenceTaskHandle = nullptr;

String species_list[5] = { "Larus michahellis", "Columba livia", "Myiopsitta monachus", "Psittacula krameri", "Corvus cornix" };

//============================== I2S pin config ==============================
static const i2s_pin_config_t pin_config =
{
    .bck_io_num    = MIC_SCK,
    .ws_io_num     = MIC_WS,
    .data_out_num  = I2S_PIN_NO_CHANGE,
    .data_in_num   = MIC_SD
};

//============================== I2S driver config ==============================
static i2s_config_t i2s_cfg =
{
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE_HZ,
    .bits_per_sample      = SAMPLE_BITS,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = DMA_BUF_COUNT_INT32,
    .dma_buf_len          = DMA_BUF_LEN,
    .use_apll             = true,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
};

/*
 * I2S event task
 *
 * FIll an audio buffer, and swap it immediately
 * NN model should account for unconsistent band data (through augmentation)
 * Trigger a spectrogram computation every HOP_BUFFERS filled audio buffers
 */
static void i2s_event_task(void* arg)
{
    i2s_event_t evt;
    size_t      bytesRead;

    while(true)
    {
        if (xQueueReceive(i2s_evt_queue, &evt, portMAX_DELAY) == pdTRUE)
        {
            //Serial.printf("I2S stack headroom: %u bytes\n", uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t)); // Print minimum ever free available task space
            if (evt.type == I2S_EVENT_RX_DONE)
            {
                // read one full DMA buffer
                i2s_read( I2S_NUM, pingPongBuf[currentBuf], DMA_BUF_LEN * sizeof(int32_t), &bytesRead, 0 );

                // trigger spectrogram every HOP_BUFFERS
                // if (SpectrogramTaskHandle && ++bufferCount >= HOP_BUFFERS) 
                // {
                //     xTaskNotifyGive(SpectrogramTaskHandle);
                //     bufferCount = 0;
                // }

                // bufIndex = (bufIndex + 1) % DMA_BUF_COUNT;

                BufferPreprocess(pingPongBuf[currentBuf]);
                currentBuf ^= 1;
            }
        }
    }
}

void BufferPreprocess(int32_t* buf)
{
    // Convert BE→LE & take top 16 bits of each 24-bit sample
    for (size_t i = 0; i < DMA_BUF_LEN; i++) 
    {
        // uint32_t w = __builtin_bswap32(buf[i]);
        // audioBufs[bufIndex][i] = (int16_t)(w >> 16);
        audioBufs[bufIndex][i] = (int16_t)(buf[i] >> 8);
    }

    // trigger spectrogram every HOP_BUFFERS
    if (SpectrogramTaskHandle && ++bufferCount >= HOP_BUFFERS) 
    {
        xTaskNotifyGive(SpectrogramTaskHandle);
        bufferCount = 0;
    }
    bufIndex = (bufIndex + 1) % DMA_BUF_COUNT;
}

/* 
 * Spectrogram task
 *
 * Initialize some variables, also free them on task end (e.g. to free some kb of memory for other tasks)
 * Compute a whole spectrogram at a time when signaled to
 * Use 2048 samples for FFT computation
 * Use a 50% hop between frames
 * Output the spectrogram in a global var in decibels
 */
static void SpectrogramTask(void* arg) 
{
    // Initialize DSP components
    hann_window = (float*)heap_caps_malloc(FFT_SIZE * sizeof(float), MALLOC_CAP_32BIT | MALLOC_CAP_8BIT);
    fft_buffer = (int16_t*)heap_caps_malloc(FFT_SIZE * 2 * sizeof(int16_t), MALLOC_CAP_32BIT | MALLOC_CAP_8BIT);
    power_spectrum = (int32_t*)heap_caps_malloc(FFT_SIZE * sizeof(int32_t), MALLOC_CAP_8BIT);

    // Check for correct initialization
    assert(hann_window != nullptr);
    assert(fft_buffer != nullptr);
    assert(power_spectrum != nullptr);

    // Generate Hann window
    dsps_wind_hann_f32(hann_window, FFT_SIZE);

    // Initialize FFT
    esp_err_t ret = dsps_fft2r_init_sc16(NULL, FFT_SIZE);
    if (ret != ESP_OK) 
    {
        Serial.printf("FFT init failed: %d\n", ret);
        vTaskDelete(NULL); // Important: Delete the task on error!
    }

    // Flag used for skipping first second of sampling
    bool bFirstCall = true;

    while (true) 
    {
        // wait for notification
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // Print minimum ever free available task space
        // Serial.printf("Spectrogram stack headroom: %u bytes\n", uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t));

        // Skip if not enough data has been collected yet
        if (bFirstCall && bufIndex < WIN_BUFFERS) 
        {
            bFirstCall = false;
            continue;
        }

        unsigned long time = micros();
        
        // Copy value to avoid concurrent read/write
        size_t bufIndex_copy = bufIndex - 1;

        /*** START OF SPECTROGRAM COMPUTATION ***/
        for (int frame_index = 0; frame_index < SPECTROGRAM_WIDTH; frame_index++) 
        {
            // Compute the buffer index for this frame
            int raw = bufIndex_copy - WIN_BUFFERS + frame_index;
            int idx = (raw % DMA_BUF_COUNT + DMA_BUF_COUNT) % DMA_BUF_COUNT;

            for (int i = 0; i < FFT_SIZE; i++) 
            {
                fft_buffer[2 * i + 0] = audioBufs[idx][i];// * hann_window[i];
                fft_buffer[2 * i + 1] = 0;
            }

            // Perform FFT
            dsps_fft2r_sc16_aes3(fft_buffer, FFT_SIZE);
            dsps_bit_rev_sc16_ansi(fft_buffer, FFT_SIZE);
            // dsps_cplx2reC_sc16(fft_buffer, FFT_SIZE);
            
            // Compute power spectrum for real FFT output
            for (int k = 0; k <= FFT_SIZE / 2; k++) 
            {
                float re = fft_buffer[k * 2 + 0];     // Re[k]
                float im = fft_buffer[k * 2 + 1];     // Im[k]
                power_spectrum[k] = re*re + im*im;
            }

            // sum into NUM_BANDS equal-width bands and log10 scale
            int binsPerBand = (FFT_SIZE / 2) / NUM_BANDS;
            for (int b = 0; b < NUM_BANDS; b++) 
            {
                float sum = 0.0f;
                for (int k = b * binsPerBand; k < (b+1) * binsPerBand; k++) 
                {
                    sum += power_spectrum[k];
                }
                band_energy = 10.0f * log10f(sum + 1e-12f);

                // Clip and scale to cover all 0 to 255 uint8 values
                band_energy = constrain(band_energy, SPECTRUM_FLOOR, SPECTRUM_CEIL);
                band_energy = (band_energy - SPECTRUM_FLOOR) * SPECTRUM_SCALE;
                band_energy = constrain(band_energy, 0.f, 255.f);
                spectrogram_data[b + frame_index * NUM_BANDS] = (uint8_t)band_energy;
            }
        }
        /*** END OF SPECTROGRAM COMPUTATION ***/

        Serial.println("[Spectrogram] Performance ms:" + String((micros() - time) / 1000.f));

        // DisplaySpectrogram_Serial();
        // DisplaySpectrogram_OLED_128bands(); 

        Display_ST7789V();
        
        xTaskNotifyGive(InferenceTaskHandle);

    } // End of while loop

    // Cleanup - This code is reached only if the task is explicitly deleted.
    free(hann_window);
    free(fft_buffer);
    free(power_spectrum);
    dsps_fft2r_deinit_sc16();
    vTaskDelete(NULL); // Delete the task itself.
}

/* 
 * Inference task
 *
 * Do a prediction and display it
 */
static void InferenceTask(void* arg) 
{
    float scores[5];
    while (true) 
    {
        // wait for notification
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        // Print minimum ever free available task space
        //Serial.printf("Inference stack headroom: %u bytes\n", uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t)); 
        
        // Save time for evaluation
        unsigned long time = micros();

        // Do a prediction
        if (tfliteModel.Inference(spectrogram_data, NUM_BANDS * SPECTROGRAM_WIDTH, scores, 5)) 
        {   
            // Display the results
            Serial.println("[Inference] Performance ms:" + String((micros() - time) / 1000.f));
            DisplayScores_OLED(scores, 5, (micros() - time) / 1000.f);
            DisplayScores_Serial(scores, 5);
        }
    }
}

/*
 * Monitor Task
 * 
 * Serially print some memory metrics
 */
void monitorTask(void* pv) {
    while(true) 
    {
        // delay 100 ms
        vTaskDelay(pdMS_TO_TICKS(1000));

        // query heaps
        size_t free8    = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        size_t largest8 = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
        size_t freeDma  = heap_caps_get_free_size(MALLOC_CAP_DMA);

        // print stats
        Serial.printf(
            "DRAM free: %6u, largest: %6u | DMA heap: %6u\n",
            (unsigned)free8,
            (unsigned)largest8,
            (unsigned)freeDma
        );
    }
}

//==================== Setup & loop ====================
void setup()
{
    Wire.begin(I2C_SDA, I2C_SCL); // SDA, SCL for pi pico communication

    Serial.begin(921600);
    delay(100);

    VextON();       // Power OLED
    display.init();
    display.clear();
    display.setContrast(255);
    display.display();

    // allocate DMA buffers
    for (int i = 0; i < DMA_BUF_COUNT; i++)
    {
        audioBufs[i] = (int16_t*) heap_caps_malloc(
            DMA_BUF_LEN * sizeof(int16_t),
            MALLOC_CAP_DMA
        );

        if (!audioBufs[i])
        {
            Serial.printf("Buffer %d alloc failed\n", i);
            while (1) { delay(1); }
        }
    }

    // Init the TFLite model - allocate tensor arena, model. etc.
    tfliteModel.Init();
    spectrogram_data = tfliteModel.GetInputTensor();
    if(spectrogram_data == nullptr)
    {
        spectrogram_data = (uint8_t*)malloc(NUM_BANDS * SPECTROGRAM_WIDTH * sizeof(uint8_t));
    }

    // start monitoring task
    // xTaskCreate( monitorTask, "TaskMonitor", 2048, nullptr, tskIDLE_PRIORITY + 1, nullptr );

    // install I2S driver with event queue
    ESP_ERROR_CHECK( i2s_driver_install( I2S_NUM, &i2s_cfg, DMA_BUF_COUNT, &i2s_evt_queue) );
    ESP_ERROR_CHECK( i2s_set_pin(I2S_NUM, &pin_config) );
    ESP_ERROR_CHECK( i2s_zero_dma_buffer(I2S_NUM) );
  
    // start inference task
    xTaskCreatePinnedToCore( InferenceTask, "Inference", 1024 * 3, nullptr, configMAX_PRIORITIES, &InferenceTaskHandle, 0 );                // stack - 2700 bytes used

    // start spectrogram task
    xTaskCreatePinnedToCore( SpectrogramTask, "Spectrogram", 1024 * 3, nullptr, configMAX_PRIORITIES - 1, &SpectrogramTaskHandle, 0 );      // stack - 1400 bytes used

    // start I2S event task
    xTaskCreatePinnedToCore( i2s_event_task, "I2S_Event", 2048, nullptr, configMAX_PRIORITIES - 1, nullptr, 1 );
}

void loop()
{
    // Useless function in FreeRTOS
}

/*
 * Displaying and serial print funcions
 */
void DisplaySpectrogram_Serial()
{
    // Output spectrogram data after processing all 16 buffers
    // after spectrogram_data[...] is filled
    Serial.println("Spectrogram (128 bands):");
    for (int i = 0; i < SPECTROGRAM_WIDTH; i++) {
        for (int j = 0; j < NUM_BANDS; j++) {
            // Serial.print(spectrogram_data[j + i * NUM_BANDS], 2);
            Serial.print(spectrogram_data[j + i * NUM_BANDS]);
            if (j+1 < NUM_BANDS) Serial.print(',');
        }
        Serial.println();
    }
}

void DisplaySpectrogram_OLED()
{
    // Clear the OLED display
    display.clear();

    const int frame_count = (WIN_BUFFERS - 1) * 2; // 30 frames
    const int display_width = 128;
    const int display_height = 64;
    const int column_spacing = 2;
    const int usable_columns = display_width / column_spacing;
    const int offset_x = (display_width - frame_count * column_spacing) / 2;

    for (int i = 0; i < frame_count; i++) 
    {
        int x = offset_x + i * column_spacing;
        if (x >= display_width) break;

        for (int j = 0; j < NUM_BANDS; j++) 
        {
            float db = spectrogram_data[j + i * NUM_BANDS];
            if (db < 50.0f) continue;

            // Scale 50–120 dB to 0–64 px
            float scaled = (db - 50.0f) / (120.0f - 50.0f);
            int height = (int)(scaled * display_height);
            if (height > display_height) height = display_height;

            int y_start = display_height - height;
            display.drawLine(x, y_start, x, display_height - 1);
        }
    }

    display.display();
}

void DisplaySpectrogram_OLED_128bands() 
{
    display.clear();
    const int W      = display.getWidth();    // 128
    const int H      = display.getHeight();   // 64
    const int bands  = NUM_BANDS;             // 64
    const int frames = (WIN_BUFFERS-1)*2;     // 30
    const int xStep  = W / bands;             // 2 px per band
    const int yStep  = H / frames;            // ~2 px per time-step

    for (int b = 0; b < bands; b++) 
    {
        int x0 = b * xStep;
        for (int f = 0; f < frames; f++) 
        {
            float db = spectrogram_data[b + f * bands];
            if (db < 50.0f) continue;
            int h = (int)((db - 50.0f) / (120.0f - 50.0f) * yStep);
            if (h > yStep) h = yStep;
            int y0 = f * yStep + (yStep - h);
            display.drawLine(x0, y0, x0, y0 + h - 1);
        }
    }
    display.display();
}

void DisplayScores_Serial(const float* scores, int numScores) {
    for (int i = 0; i < numScores; i++) 
    {
        Serial.print("Score["); Serial.print(i); Serial.print("] = ");
        Serial.println(scores[i], 6);
    }
}

void DisplayScores_OLED(const float* scores, int numScores, float ms) 
{
    display.clear();
    
    display.setTextAlignment(TEXT_ALIGN_LEFT);
    display.setFont(ArialMT_Plain_10);
    
    // Print each score on its own line
    int x = 0;
    int y = 0;
    for (int i = 0; i < numScores; i++) 
    {
        String line = "Score[" + String(i) + "]: " + String(scores[i], 2) + " " + species_list[i];
        display.drawString(0, y, line);
        y += 10;
    }
    
    display.setTextAlignment(TEXT_ALIGN_RIGHT);
    display.setFont(ArialMT_Plain_10);
    x = display.width();
    y = display.height()-12;
    display.drawString(x, y, String(ms));

    display.display();
}

// Stream a big buffer in CHUNK_SIZE slices
void SendSpectrogram() 
{
    const size_t CHUNK_SIZE = 128;
    const size_t length = NUM_BANDS * SPECTROGRAM_WIDTH;
    for (size_t offset = 0; offset < length; offset += CHUNK_SIZE) 
    {
        size_t this_len = min(CHUNK_SIZE, length - offset);
        Wire.beginTransmission(SLAVE_ADDR);
        Wire.write(spectrogram_data + offset, this_len);
        Wire.endTransmission();
        delay(1);  // give the Pico time to store it
    }
}

void Display_ST7789V()
{
    Wire.beginTransmission(SLAVE_ADDR);
    uint32_t cmd = 0x00FEED00;
    uint8_t buf[5] = {
        uint8_t(cmd>>24), 
        uint8_t(cmd>>16),
        uint8_t(cmd>> 8), 
        uint8_t(cmd    ),
        0
    };
    
    buf[4] = buf[0]+buf[1]+buf[2]+buf[3];
    Wire.write(buf, 5);
    Wire.endTransmission();
    
    SendSpectrogram();
}

/*
 * Utils
 *
 * Just to enable the SD1306 OLED display on the Heltec LoRa V3
 */
void VextON(void)
{
  pinMode(Vext,OUTPUT);
  digitalWrite(Vext, LOW);
}

void VextOFF(void) //Vext default OFF
{
  pinMode(Vext,OUTPUT);
  digitalWrite(Vext, HIGH);
}
