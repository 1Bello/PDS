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
#include "esp_now.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_mac.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_main.h"
#include "esp_psram.h"

#define INF_POWER 0.13
#define ESP_CHANNEL 1

//static uint8_t peer_mac [ESP_NOW_ETH_ALEN] = {0x3c, 0x71, 0xbf, 0xef, 0x67, 0xd0};
static uint8_t peer_mac [ESP_NOW_ETH_ALEN] = {0xfc, 0xe8, 0xc0, 0xce, 0x53, 0xd4};
// 3c:71:bf:ef:67:d0
//ESP-NOW
static const char * TAG = "esp_now_init";
static esp_err_t init_wifi(void)
{
    wifi_init_config_t wifi_init_config = WIFI_INIT_CONFIG_DEFAULT();

    esp_netif_init();
    esp_event_loop_create_default();
    nvs_flash_init();
    esp_wifi_init(&wifi_init_config);
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_storage(WIFI_STORAGE_FLASH);
    esp_wifi_start();

    ESP_LOGI(TAG, "wifi init completed");
    return ESP_OK;  
}

void recv_cb(const esp_now_recv_info_t * esp_now_info, const uint8_t *data, int data_len)
{
    ESP_LOGI(TAG, "Data Received" MACSTR " %s", MAC2STR(esp_now_info->src_addr), data);
}

void send_cb(const uint8_t *mac_addr, esp_now_send_status_t status)
{
    if (status == ESP_NOW_SEND_SUCCESS)
    {
          ESP_LOGI(TAG, "ESP_NOW_SEND_SUCCESS");
    }
    else{
          ESP_LOGW(TAG, "ESP_NOW_SEND_FAIL");
    }
}

static esp_err_t init_esp_now(void)
{
    esp_now_init();
    esp_now_register_recv_cb(recv_cb);
    esp_now_register_send_cb(send_cb);

    ESP_LOGI(TAG, "esp now init completed");
    return ESP_OK;
}

static esp_err_t register_peer(uint8_t *peer_addr)
{
    esp_now_peer_info_t esp_now_peer_info = {};
    memcpy(esp_now_peer_info.peer_addr, peer_mac, ESP_NOW_ETH_ALEN);
    esp_now_peer_info.channel = ESP_CHANNEL;

    esp_now_add_peer(&esp_now_peer_info);
    return ESP_OK;
}

static esp_err_t esp_now_send_data(const uint8_t *peer_addr, const uint8_t *data, uint8_t len)
{
    esp_now_send(peer_addr, data, len );
    return ESP_OK;
}




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

  //ESP-NOW
  ESP_ERROR_CHECK(init_wifi());
  ESP_ERROR_CHECK(init_esp_now());
  ESP_ERROR_CHECK(register_peer(peer_mac));

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

  int max_score_index = 0;
  float max_score = sign_scores[0];
  for (int i = 1; i < kCategoryCount; ++i) {
    if (sign_scores[i] > max_score) {
      max_score = sign_scores[i];
      max_score_index = i;
    }
  }
  if(max_score_index == 0){
    esp_now_send_data(peer_mac, (uint8_t*) 'A', 32);
  }
  if(max_score_index == 1){
    esp_now_send_data(peer_mac, (uint8_t*) 'B', 32);
  }
  if(max_score_index == 2){
    esp_now_send_data(peer_mac, (uint8_t*) 'C', 32);
  }
  if(max_score_index == 3){
    esp_now_send_data(peer_mac, (uint8_t*) 'D', 32);
  }
  if(max_score_index == 4){
    esp_now_send_data(peer_mac, (uint8_t*) 'E', 32);
  }
  if(max_score_index == 5){
    esp_now_send_data(peer_mac, (uint8_t*) 'F', 32);
  }
  

  
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
