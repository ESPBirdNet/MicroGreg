#include <Arduino.h>
#include "model.h"
#include "TFLiteModel.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace 
{
  const tflite::Model * model = nullptr;
  tflite::MicroInterpreter * interpreter = nullptr;
  TfLiteTensor * input_tensor = nullptr;
  TfLiteTensor * output_tensor = nullptr;
  int inference_count = 0;

  constexpr int kTensorArenaSize = 128 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


/*
 *  Class constructor
 */
TFLiteModel::TFLiteModel()
{
    // Do nothing here, just allocate object memory
}

/*
 * Inits - to be called
 */
void TFLiteModel::Init()
{
    if(Serial.available())
    {
        Serial.println("[TFLite] Initializing and loading TFLite model!");
    }
    TFLiteModel_Init();
}

// Init the interpreter with the needed activations kernels
void TFLiteModel::Interpreter_Init() 
{
    // Increase the number of registered operations as needed
    static tflite::MicroMutableOpResolver<13> resolver;

    // Register necessary operations
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSub();
    resolver.AddDiv();
    resolver.AddMean();
    resolver.AddRelu();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
}

// Allocate necessary resources, init interpreter, setup input/output tensors
void TFLiteModel::TFLiteModel_Init()
{   
    // Load model from binary model in TFLite.h
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) 
    {
        Serial.println("Model version mismatch! Check your .tflite file.");
        while (1);
    }

    Interpreter_Init();
   
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) 
    {
      MicroPrintf("AllocateTensors() failed");
      return;
    }

    // Obtain pointers to the model's input and output tensors.
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    // Keep track of how many inferences we have performed.
    inference_count = 0;
}

/*
 * Do a prediction
 * To use with full float32 models
 * Inference will be slower, memory usage higher
 */
bool TFLiteModel::Inference(float * input_data, int input_size, float * output_data, int output_size)
{
    // Sanity check: does the input tensor have enough space?
    if (input_size != input_tensor->bytes / sizeof(float)) 
    {
        Serial.print("Expected ");
        Serial.print(input_tensor->bytes);
        Serial.print(" floats, got ");
        Serial.println(input_size);
        return false;
    }

    for (int i = 0; i < input_size; i++) 
    {
        input_tensor->data.f[i] = input_data[i];
    }

    // Copy input_data into the tensor
    // memcpy(input_tensor->data.f, input_data, input_size * sizeof(float));

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) 
    {
        Serial.println("Inference failed");
        return false;
    }

    for (int i = 0; i < output_size; i++) 
    {
        output_data[i] = output_tensor->data.f[i];
    }

    inference_count++;

    return true;
}

/*
 * Do a prediction
 * Quantize the input array
 * Ensure that the model quantization parameters are well calibrated 
 */
bool TFLiteModel::Inference_Quantize(float * input_data, int input_size, float * output_data, int output_size)
{
    // Sanity check: does the input tensor have enough space?
    if (input_size != input_tensor->bytes) 
    {
        Serial.print("Expected ");
        Serial.print(input_tensor->bytes);
        Serial.print(" floats, got ");
        Serial.println(input_size);
        return false;
    }

    // Quantize input
    const float   scale      = input_tensor->params.scale;
    const int32_t zero_point = input_tensor->params.zero_point;
    for (int i = 0; i < input_size; i++) 
    {
        // 1) scale and shift
        int32_t q = lrintf(input_data[i] / scale) + zero_point;
        // 2) clip to tensor’s valid range
        if (input_tensor->type == kTfLiteUInt8) 
        {
            q = q < 0   ? 0   : (q > 255 ? 255 : q);
            input_tensor->data.uint8[i] = static_cast<uint8_t>(q);
        } 
        else 
        {  // kTfLiteInt8
            q = q < -128 ? -128 : (q > 127 ? 127 : q);
            input_tensor->data.int8[i] = static_cast<int8_t>(q);
        }
    }

    // Copy input_data into the tensor
    // memcpy(input_tensor->data.f, input_data, input_size * sizeof(float));

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) 
    {
        Serial.println("Inference failed");
        return false;
    }

    // De‑quantize output bytes to floats
    const float   out_scale      = output_tensor->params.scale;
    const int32_t out_zero_point = output_tensor->params.zero_point;
    const int     n_out = output_tensor->bytes;  // number of output elements
    const int     to_copy = min(n_out, output_size);
    if (output_tensor->type == kTfLiteUInt8) 
    {
        uint8_t *src = output_tensor->data.uint8;
        for (int i = 0; i < to_copy; i++) 
        {
            output_data[i] = (src[i] - out_zero_point) * out_scale;
        }
    } 
    else 
    {  // kTfLiteInt8
        int8_t *src = output_tensor->data.int8;
        for (int i = 0; i < to_copy; i++) 
        {
            output_data[i] = (src[i] - out_zero_point) * out_scale;
        }
    }

    inference_count++;

    return true;
}