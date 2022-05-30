// are you using an I2S microphone - comment this out if you want to use an analog mic and ADC input
#define USE_I2S_MIC_INPUT

// Which channel is the I2S microphone on? I2S_CHANNEL_FMT_ONLY_LEFT or I2S_CHANNEL_FMT_ONLY_RIGHT
#define I2S_MIC_CHANNEL I2S_CHANNEL_FMT_ONLY_LEFT
// #define I2S_MIC_CHANNEL I2S_CHANNEL_FMT_ONLY_RIGHT
#define I2S_MIC_SERIAL_CLOCK GPIO_NUM_33
#define I2S_MIC_LEFT_RIGHT_CLOCK GPIO_NUM_26
#define I2S_MIC_SERIAL_DATA GPIO_NUM_25

#define RXp2 16
#define TXp2 17

#define WIFI_SSID "TechPublic"
#define WIFI_PASSWORD ""

#define API_KEY "AIzaSyAonx30i-i5ww0ftTMvQYrkc8eX2J1DOx4"
#define DATABASE_URL "https://voice-control-44c54-default-rtdb.firebaseio.com/" 