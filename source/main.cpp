#include "ei_run_classifier.h"

#include "imu_provider.h"
#include "rasterize_stroke.h"

#include <hardware/gpio.h>
#include <hardware/irq.h>
#include <hardware/uart.h>
#include <pico/stdio_usb.h>
#include <stdio.h>
#include "LCD_st7735.h"

#define UART_ID uart0
#define BAUD_RATE 115200
#define DATA_BITS 8
#define STOP_BITS 1
#define PARITY UART_PARITY_NONE
#define UART_TX_PIN 0
#define UART_RX_PIN 1

const uint LED_PIN = 25;

// Constants for image rasterization
constexpr int raster_width = 32;
constexpr int raster_height = 32;
constexpr int raster_channels = 3;
constexpr int raster_byte_count =
    raster_height * raster_width * raster_channels;
int8_t raster_buffer[raster_byte_count];

bool linked = false;
bool first = true;
uint16_t send_index = 0;

#ifndef DO_NOT_OUTPUT_TO_UART
// RX interrupt handler
uint8_t command[32] = {0};
bool start_flag = false;
int receive_index = 0;
uint8_t previous_ch = 0;

void on_uart_rx() {
  uint8_t current_ch = 0;
  while (uart_is_readable(UART_ID)) {
    current_ch = uart_getc(UART_ID);
    //    printf("%c \n", current_ch);
    if (start_flag) {
      command[receive_index++] = current_ch;
    }
    if (current_ch == 0xf4 && previous_ch == 0xf5) {
      start_flag = true;
    } else if (current_ch == 0x0a && previous_ch == 0x0d) {
      start_flag = false;
      // add terminator
      command[receive_index - 2] = '\0';

      receive_index = 0;
      if (strcmp("IND=BLECONNECTED", (const char *)command) == 0) {
        linked = true;
      } else if (strcmp("IND=BLEDISCONNECTED", (const char *)command) == 0) {
        linked = false;
      }
      printf("%s\n", command);
    }
    previous_ch = current_ch;
  }
}

void setup_uart() {
  // Set up our UART with the required speed.
  uint baud = uart_init(UART_ID, BAUD_RATE);
  // Set the TX and RX pins by using the function select on the GPIO
  // Set datasheet for more information on function select
  gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
  gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);
  // Set our data format
  uart_set_format(UART_ID, DATA_BITS, STOP_BITS, PARITY);
  // Turn off FIFO's - we want to do this character by character
  uart_set_fifo_enabled(UART_ID, false);
  // Set up a RX interrupt
  // We need to set up the handler first
  // Select correct interrupt for the UART we are using
  int UART_IRQ = UART_ID == uart0 ? UART0_IRQ : UART1_IRQ;

  // And set up and enable the interrupt handlers
  irq_set_exclusive_handler(UART_IRQ, on_uart_rx);
  irq_set_enabled(UART_IRQ, true);

  // Now enable the UART to send interrupts - RX only
  uart_set_irq_enables(UART_ID, true, false);
}
#else
void setup_uart() {}
#endif

int main() {
  stdio_usb_init();
  setup_uart();

  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);

  gpio_put(LED_PIN, !gpio_get(LED_PIN));

  ST7735_Init();
  ST7735_DrawImage(0, 0, 80, 160, arducam_logo);

  SetupIMU();

  ei_impulse_result_t result = {nullptr};

  gpio_put(LED_PIN, !gpio_get(LED_PIN));

  ST7735_FillScreen(ST7735_GREEN);

  ST7735_DrawImage(0,0,80,40,(uint8_t*)IMU_ICM20948);

  ST7735_WriteString(5, 45, "Magic", Font_11x18, ST7735_BLACK, ST7735_GREEN);
  ST7735_WriteString(30, 70, "Wand", Font_11x18, ST7735_BLACK, ST7735_GREEN);

  while (true) {
    gpio_put(LED_PIN, !gpio_get(LED_PIN));
    int accelerometer_samples_read;
    int gyroscope_samples_read;

    ReadAccelerometerAndGyroscope(&accelerometer_samples_read,
                                  &gyroscope_samples_read);

    // Parse and process IMU data
    bool done_just_triggered = false;
    if (gyroscope_samples_read > 0) {
      EstimateGyroscopeDrift(current_gyroscope_drift);
      UpdateOrientation(gyroscope_samples_read, current_gravity,
                        current_gyroscope_drift);
      UpdateStroke(gyroscope_samples_read, &done_just_triggered);
      if (linked) {
        if (first) {
          first = false;
        }
        if (send_index++ % 16 == 0) {
          uart_write_blocking(UART_ID, stroke_struct_buffer, 328);
        }
      } else {
        first = true;
        send_index = 0;
      }
    }

    if (accelerometer_samples_read > 0) {
      // The accelerometer data is read and passed to the
      // EstimateGravityDirection() where it is used to determine the
      // orientation of the Arduino with respect to the ground.
      EstimateGravityDirection(current_gravity);
      // The Accelerometer data is passed to UpdateVelocity() where it is used
      // to calculate the velocity of the Arduino.
      UpdateVelocity(accelerometer_samples_read, current_gravity);
    }

    if (done_just_triggered and !linked) {
      // Rasterize the gesture
      RasterizeStroke(stroke_points, *stroke_transmit_length, 0.6f, 0.6f,
                      raster_width, raster_height, raster_buffer);
      for (int y = 0; y < raster_height; ++y) {
        char line[raster_width + 1];
        for (int x = 0; x < raster_width; ++x) {
          const int8_t *pixel =
              &raster_buffer[(y * raster_width * raster_channels) +
                             (x * raster_channels)];
          const int8_t red = pixel[0];
          const int8_t green = pixel[1];
          const int8_t blue = pixel[2];
          char output;
          if ((red > -128) || (green > -128) || (blue > -128)) {
            output = '#';
          } else {
            output = '.';
          }
          line[x] = output;
        }
        line[raster_width] = 0;
        ei_printf("%s\n", line);
      }
      matrix_t features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

      for (int i = 0; i < EI_CLASSIFIER_NN_INPUT_FRAME_SIZE; i++) {
        float tmp = ((float)raster_buffer[i] + 128.0) / 255.0;
        features_matrix.buffer[i] = tmp;
      }
      ei_printf("\n");

      ei_printf("Edge Impulse standalone inferencing (Raspberry Pico 2040)\n");

      // invoke the impulse
      EI_IMPULSE_ERROR res = run_inference(&features_matrix, &result, false);
      ;
      ei_printf("run_classifier returned: %d\n", res);

      if (res != 0)
        return res;

      // print the predictions
      ei_printf("Predictions ");
      ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
                result.timing.dsp, result.timing.classification,
                result.timing.anomaly);
      ei_printf(": \n");

      // human-readable predictions
      float max_score=0.0f;
      const char *max_index;
      for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        if (ix!=0 && (ix)%5==0){
          ei_printf("\n");
        }
        ei_printf("%s: %.2f\t", result.classification[ix].label, result.classification[ix].value);
        const float score = result.classification[ix].value;
        if ((ix == 0) || (score > max_score)) {
          max_score = score;
          max_index = ei_classifier_inferencing_categories[ix];
        }
      }
      ei_printf("\nFound %s (%0.2f)\n",max_index,max_score*100);
      char str[10];
      sprintf(str, "%d%%", (int )(max_score*100));

      ST7735_FillRectangle(0, 90, ST7735_WIDTH, 160 - 90, ST7735_GREEN);
      ST7735_WriteString(35, 100, max_index, Font_11x18, ST7735_BLACK,
                         ST7735_GREEN);
      ST7735_WriteString(25, 130, str, Font_11x18, ST7735_BLACK, ST7735_GREEN);


#if EI_CLASSIFIER_HAS_ANOMALY == 1
      ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif
    }
  }
}