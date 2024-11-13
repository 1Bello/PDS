#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
#include "esp_psram.h"

#define INF_POWER 0.13

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 40 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif

#ifndef portTICK_RATE_MS
#define portTICK_RATE_MS portTICK_PERIOD_MS
#endif

static int kTensorArenaSize = 176 * 1024 + scratchBufSize;  // Reduced size for testing
static uint8_t *tensor_arena;
}  // namespace

void setup() {
  // Initialize PSRAM
  if (esp_psram_get_size() == 0) {
    printf("PSRAM not found\n");
    return;
  }

  printf("Total heap size: %d\n", heap_caps_get_total_size(MALLOC_CAP_8BIT));
  printf("Free heap size: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
  printf("Total PSRAM size: %d\n", esp_psram_get_size());
  printf("Free PSRAM size: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Initialize model
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Allocate tensor arena in PSRAM
  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }
  printf("Free heap size after allocation: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
  printf("Free PSRAM size after allocation: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Define MicroMutableOpResolver and add required operations
  static tflite::MicroMutableOpResolver<7> micro_op_resolver;
  micro_op_resolver.AddQuantize(); 
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddDequantize();

  // Build and allocate the interpreter
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
void loop() {

  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.f)) {
    MicroPrintf("Image capture failed.");
  }

  // for (int i = 0; i < kNumCols * kNumRows; i++) {
  //   printf("%f, ", input->data.f[i]);
  // }
  // printf("\n");

  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  printf("Input type: %s\n", TfLiteTypeGetName(input->type));
  printf("Output type: %s\n", TfLiteTypeGetName(output->type));

  float sign_scores[kCategoryCount];
  for (int i = 0; i < kCategoryCount; ++i) {
    sign_scores[i] = output->data.f[i];
  }
  RespondToDetection(sign_scores, kCategoryLabels);
  vTaskDelay(7000 / portTICK_RATE_MS);
}
#endif

#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long act_total_time;
  extern long long q_total_time;
  extern long long conv_total_time;
  extern long long pooling_total_time;
  extern long long resh_total_time;
  extern long long fc_total_time;
  extern long long softmax_total_time;
  extern long long dq_total_time;
#endif

#ifdef CLI_ONLY_INFERENCE
void run_inference(void *ptr) {
  /* Convert from uint8 picture data to float */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
      input->data.f[i] = ((float*) ptr)[i];
      // printf("%f, ", input->data.f[i]);
  }
  // printf("\n");

#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Quantize time = %.3f [ms]\n", q_total_time / 1000.0);
  printf("Conv2D total time = %.3f [ms]\n", conv_total_time / 1000.0);
  printf("MaxPool2D total time = %.3f [ms]\n", pooling_total_time / 1000.0);
  printf("Reshape time = %.3f [ms]\n", resh_total_time / 1000.0);
  printf("FullyConnected total time = %.3f [ms]\n", fc_total_time / 1000.0);
  printf("Softmax time = %.3f [ms]\n", softmax_total_time / 1000.0);
  printf("Dequantize time = %.3f [ms]\n", dq_total_time / 1000.0);
  printf("Total time = %.3f [ms]\n\n", total_time / 1000.0);

  printf("Quantize energy = %f [J]\n", INF_POWER * (q_total_time / 1000000.0));
  printf("Conv2D energy = %f [J]\n", INF_POWER * (conv_total_time / 1000000.0));
  printf("MaxPool2D energy = %f [J]\n", INF_POWER * (pooling_total_time / 1000000.0));
  printf("Reshape energy = %f [J]\n", INF_POWER * (resh_total_time / 1000000.0));
  printf("FullyConnected energy = %f [J]\n", INF_POWER * (fc_total_time / 1000000.0));
  printf("Softmax energy = %f [J]\n", INF_POWER * (softmax_total_time / 1000000.0));
  printf("Dequantize energy = %f [J]\n", INF_POWER * (dq_total_time / 1000000.0));
  printf("Total energy = %f [J]\n\n", INF_POWER * (total_time / 1000000.0));

  /* Reset times */
  total_time = 0;
  act_total_time = 0;
  q_total_time = 0;
  conv_total_time = 0;
  pooling_total_time = 0;
  resh_total_time = 0;
  fc_total_time = 0;
  softmax_total_time = 0;
  dq_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);

  // printf("Input type: %s\n", TfLiteTypeGetName(input->type));
  // printf("Output type: %s\n", TfLiteTypeGetName(output->type));

  float sign_scores[kCategoryCount];
  for (int i = 0; i < kCategoryCount; ++i) {
    sign_scores[i] = output->data.f[i];
  }
  RespondToDetection(sign_scores, kCategoryLabels);
}
#endif
