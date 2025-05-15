#ifndef TFLiteModel_h
#define TFLiteModel_h

#include <Arduino.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

class TFLiteModel
{
    public:
        TFLiteModel();
        // ~TFLiteModel();

        void Init();
        bool Inference(uint8_t * input_data, int input_size, float * output_data, int output_size);

    private:
        void TFLiteModel_Init();
        void Interpreter_Init();
};

#endif