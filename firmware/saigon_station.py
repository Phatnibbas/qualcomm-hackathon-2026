# main.py
# ESP32-WROOM (ESP32 classic) - MicroPython
# Modbus Environmental Sensor -> ThingSpeak
#
# ThingSpeak channel:
# Channel ID: 3428136
#
# Field 1: WindSpeed
# Field 2: WindDirection
# Field 3: Temperature
# Field 4: Pressure
# Field 5: Light
# Field 6: Humidity
# Field 7: Noise
# Field 8: PM2.5

import gc
import network

from machine import UART
from time import sleep_ms, sleep_us, ticks_ms, ticks_diff

try:
    import requests
except ImportError:
    import urequests as requests


# =========================================================
# WIFI CONFIG
# =========================================================

# Wi-Fi và ThingSpeak Write API Key nằm ở station_secrets.py, KHÔNG nằm ở file này.
# File này thường được chia sẻ / dán vào chat, nên không được chứa bí mật.
# => Phải upload CẢ HAI file lên board: siuuuu.py và station_secrets.py
try:
    from station_secrets import (
        WIFI_SSID,
        WIFI_PASSWORD,
        THINGSPEAK_WRITE_API_KEY,
    )
except ImportError:
    print()
    print("!" * 52)
    print("THIEU station_secrets.py - hay upload file do len board.")
    print("Board se KHONG ket noi Wi-Fi va KHONG upload duoc.")
    print("!" * 52)
    print()

    WIFI_SSID = "CHUA_CAU_HINH"
    WIFI_PASSWORD = ""
    THINGSPEAK_WRITE_API_KEY = "NHAP_WRITE_API_KEY"

WIFI_RETRIES = 10
WIFI_TIMEOUT_MS = 10000


# =========================================================
# THINGSPEAK CONFIG
# =========================================================

CHANNEL_ID = 3428136

# THINGSPEAK_WRITE_API_KEY được import từ station_secrets.py ở phần trên.
# Lấy key tại: ThingSpeak -> Channels -> chọn channel -> API Keys -> Write API Key

THINGSPEAK_URL = "https://api.thingspeak.com/update"

# ThingSpeak Free yêu cầu tối thiểu 15 giây.
# Dùng 20 giây để có khoảng an toàn.
UPLOAD_INTERVAL_MS = 20000

# Chu kỳ đọc cảm biến
SENSOR_INTERVAL_MS = 5000


# =========================================================
# UART CONFIG
# =========================================================

RX_PIN = 16
TX_PIN = 17

uart = UART(
    1,
    baudrate=4800,
    bits=8,
    parity=None,
    stop=1,
    rx=RX_PIN,
    tx=TX_PIN,
    timeout=1000,
    timeout_char=20,
)


# =========================================================
# SETTINGS
# =========================================================

READ_WIND = True
DEBUG_WIND = False

# Dưới ngưỡng này coi là lặng gió, không hiện hướng.
# 0.5 m/s = start wind speed theo datasheet SEN0658. Giữ khớp với dashboard.
CALM_MS = 0.5


# =========================================================
# LCD 16x2 I2C (tùy chọn)
# =========================================================

LCD_ENABLED = True

# Đấu dây thực tế trên trạm: SDA = 22, SCL = 21.
# (Mặc định của ESP32-WROOM là ngược lại, SDA 21 / SCL 22, nên dễ nhầm.)
# Nếu sai thứ tự, lcd_init() sẽ tự thử cặp đảo và in ra cặp nào chạy được.
LCD_SDA_PIN = 22
LCD_SCL_PIN = 21

# 0 = tự dò (0x27 cho PCF8574, 0x3F cho PCF8574A)
LCD_ADDR = 0

LCD_COLS = 16

# DFRobot: "Host polling interval and waiting response time are too short,
# both need to be set above 200ms" - đây là nguyên nhân số 4 gây mất phản hồi
# trong mục troubleshooting của SEN0658. Trước đây khoảng cách chỉ 20 ms.
MODBUS_GAP_MS = 220


# =========================================================
# MODBUS COMMANDS
# =========================================================

# Wind speed and direction
COM_WIND = bytes([
    0x01, 0x03, 0x01, 0xF4,
    0x00, 0x04, 0x04, 0x07
])

# Temperature, humidity, noise
COM_THN = bytes([
    0x01, 0x03, 0x01, 0xF8,
    0x00, 0x03, 0x85, 0xC6
])

# Light
COM_LUX = bytes([
    0x01, 0x03, 0x01, 0xFE,
    0x00, 0x02, 0xA4, 0x07
])

# PM2.5, PM10, atmospheric pressure
COM_PM = bytes([
    0x01, 0x03, 0x01, 0xFB,
    0x00, 0x03, 0x75, 0xC6
])


# =========================================================
# WIFI FUNCTIONS
# =========================================================

wlan = network.WLAN(network.STA_IF)


def connect_wifi():
    """
    Kết nối Wi-Fi tối đa WIFI_RETRIES lần.
    Trả về True nếu kết nối thành công.
    """

    wlan.active(True)

    if wlan.isconnected():
        print("Wi-Fi already connected")
        print("IP:", wlan.ifconfig()[0])
        return True

    print("Connecting to Wi-Fi:", WIFI_SSID)

    for attempt in range(1, WIFI_RETRIES + 1):
        print(
            "Wi-Fi attempt {}/{}".format(
                attempt,
                WIFI_RETRIES
            )
        )

        try:
            wlan.disconnect()
        except Exception:
            pass

        sleep_ms(200)

        try:
            wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        except Exception as error:
            print("Wi-Fi connect error:", error)
            sleep_ms(1000)
            continue

        start = ticks_ms()

        while ticks_diff(ticks_ms(), start) < WIFI_TIMEOUT_MS:
            if wlan.isconnected():
                network_info = wlan.ifconfig()

                print("Wi-Fi connected")
                print("IP address:", network_info[0])
                print("Subnet:", network_info[1])
                print("Gateway:", network_info[2])
                print("DNS:", network_info[3])
                print()

                return True

            sleep_ms(500)

        print("Connection timed out, status:", wlan.status())

    print("Could not connect to Wi-Fi")
    return False


def ensure_wifi():
    if wlan.isconnected():
        return True

    print("Wi-Fi disconnected. Reconnecting...")
    return connect_wifi()


# =========================================================
# MODBUS HELPER FUNCTIONS
# =========================================================

def clear_uart():
    while uart.any():
        uart.read()
        sleep_ms(2)


def hex_bytes(data):
    """
    Đổi bytes thành chuỗi hex cách nhau bởi dấu cách.

    Không dùng data.hex(" ") vì tham số dấu phân cách không có ở mọi bản
    MicroPython. Các chỗ gọi đều nằm trong nhánh xử lý lỗi, mà nhánh lỗi
    lại là chỗ tuyệt đối không được ném thêm exception.
    """

    return " ".join("{:02x}".format(b) for b in data)


def crc16_modbus(data):
    crc = 0xFFFF

    for b in data:
        crc ^= b

        for _ in range(8):
            if crc & 0x0001:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1

    return crc & 0xFFFF


def read_n(length, timeout_ms=1000):
    buf = bytearray()
    start = ticks_ms()

    while len(buf) < length:
        if uart.any():
            chunk = uart.read(length - len(buf))

            if chunk:
                buf.extend(chunk)

        if ticks_diff(ticks_ms(), start) > timeout_ms:
            break

        sleep_ms(1)

    return bytes(buf)


def u16_be(data, index):
    return (
        (data[index] << 8)
        | data[index + 1]
    )


def s16_be(data, index):
    """
    Signed 16-bit big-endian.

    Theo protocol SEN0658, nhiệt độ dưới 0 độ C được gửi ở dạng bù hai.
    Ví dụ 0xFF9B = -101 => -10.1 độ C.
    Đọc bằng u16_be sẽ cho 65435 => 6543.5 độ C.
    """

    value = u16_be(data, index)

    if value & 0x8000:
        return value - 0x10000

    return value


def u32_be(data, index):
    return (
        (data[index] << 24)
        | (data[index + 1] << 16)
        | (data[index + 2] << 8)
        | data[index + 3]
    )


def read_modbus_frame(
    request,
    expected_byte_count,
    name="sensor",
    timeout_ms=1000
):
    """
    Modbus response:

    ADDRESS
    FUNCTION
    BYTE_COUNT
    DATA...
    CRC_LOW
    CRC_HIGH
    """

    clear_uart()
    sleep_ms(MODBUS_GAP_MS)

    uart.write(request)

    expected_len = 3 + expected_byte_count + 2
    frame = read_n(expected_len, timeout_ms)

    if len(frame) != expected_len:
        print("Time out:", name)

        if frame:
            print("Received {} bytes: {}".format(
                len(frame),
                hex_bytes(frame)
            ))
        else:
            print("Received nothing")

        return None

    if frame[0] != 0x01:
        print(
            "Wrong sensor address:",
            name,
            hex_bytes(frame)
        )
        return None

    if frame[1] != 0x03:
        print(
            "Wrong function code:",
            name,
            hex_bytes(frame)
        )
        return None

    if frame[2] != expected_byte_count:
        print(
            "Wrong byte count:",
            name,
            hex_bytes(frame)
        )
        return None

    received_crc = frame[-2] | (frame[-1] << 8)
    calculated_crc = crc16_modbus(frame[:-2])

    if received_crc != calculated_crc:
        print("CRC error:", name)
        print("RX   :", hex_bytes(frame))
        print("Got  :", hex(received_crc))
        print("Calc :", hex(calculated_crc))
        return None

    return frame


# =========================================================
# SENSOR READ FUNCTIONS
# =========================================================

def read_temperature_humidity_noise():
    frame = read_modbus_frame(
        COM_THN,
        expected_byte_count=6,
        name="temperature/humidity/noise"
    )

    if frame is None:
        return None

    data = frame[3:-2]

    humidity = u16_be(data, 0) / 10.0
    temperature = s16_be(data, 2) / 10.0
    noise = u16_be(data, 4) / 10.0

    return temperature, humidity, noise


def read_light():
    frame = read_modbus_frame(
        COM_LUX,
        expected_byte_count=4,
        name="light"
    )

    if frame is None:
        return None

    data = frame[3:-2]
    lux = u32_be(data, 0)

    return lux


def read_pm_pressure():
    frame = read_modbus_frame(
        COM_PM,
        expected_byte_count=6,
        name="PM/pressure"
    )

    if frame is None:
        return None

    data = frame[3:-2]

    pm2_5 = u16_be(data, 0)
    pm10 = u16_be(data, 2)
    pressure = u16_be(data, 4) / 10.0

    return pm2_5, pm10, pressure


def read_wind():
    frame = read_modbus_frame(
        COM_WIND,
        expected_byte_count=8,
        name="wind"
    )

    if frame is None:
        return None

    if DEBUG_WIND:
        print("WIND RAW:", hex_bytes(frame))

    data = frame[3:-2]

    r0 = u16_be(data, 0)
    r1 = u16_be(data, 2)
    r2 = u16_be(data, 4)
    r3 = u16_be(data, 6)

    if DEBUG_WIND:
        print("WIND REG:", r0, r1, r2, r3)

    wind_speed = r0 / 10.0

    # Mã hướng gió từ cảm biến
    wind_direction = r2

    # Góc hướng gió, thường từ 0 đến 360 độ
    wind_angle = r3

    return wind_speed, wind_direction, wind_angle


# =========================================================
# THINGSPEAK FUNCTIONS
# =========================================================

def format_value(value):
    """
    Chuyển số thành chuỗi trước khi gửi.
    Giới hạn số float ở 2 chữ số thập phân.
    """

    if isinstance(value, float):
        return "{:.2f}".format(value)

    return str(value)


def build_status(failed_reads, attempted, is_first_upload):
    """
    Chuỗi trạng thái gửi kèm mỗi lần upload, qua trường `status` của ThingSpeak.

    `status` KHÔNG chiếm field nào trong 8 field đang dùng, và ThingSpeak vẫn tạo
    entry khi chỉ có status. Nhờ vậy kênh vẫn có nhịp tim ngay cả khi TẤT CẢ lệnh
    Modbus timeout - phân biệt được "cảm biến chết, board còn sống" với "board chết".

    Chỉ dùng chữ cái, số, dấu chấm, gạch ngang và gạch dưới, nên luôn an toàn khi
    form-encode mà không cần hàm urlencode.
    """

    if not failed_reads:
        status = "ALL_OK"
    elif len(failed_reads) >= attempted:
        status = "FAIL-ALL"
    else:
        status = "FAIL-" + "-".join(failed_reads)

    if is_first_upload:
        status = "BOOT." + status

    return status


def send_to_thingspeak(field_values, status=""):
    """
    field_values có dạng:

    {
        "field1": 1.2,
        "field2": 180,
        "field3": 28.5
    }
    """

    # Không còn bỏ qua upload khi mọi lệnh đọc đều fail: vẫn gửi status làm nhịp tim,
    # nếu không kênh sẽ im lặng và dashboard tưởng nhầm là board mất điện.
    if not field_values and not status:
        print("No sensor data and no status to upload")
        return False

    if not ensure_wifi():
        print("Upload cancelled: no Wi-Fi")
        return False

    if (
        not THINGSPEAK_WRITE_API_KEY
        or THINGSPEAK_WRITE_API_KEY == "NHAP_WRITE_API_KEY"
    ):
        print("ThingSpeak Write API Key has not been configured")
        return False

    form_items = [
        "api_key={}".format(
            THINGSPEAK_WRITE_API_KEY
        )
    ]

    for field_name in sorted(field_values):
        value = field_values[field_name]

        if value is not None:
            form_items.append(
                "{}={}".format(
                    field_name,
                    format_value(value)
                )
            )

    if status:
        form_items.append("status={}".format(status))

    payload = "&".join(form_items)

    response = None

    try:
        print("Sending data to ThingSpeak...")

        response = requests.post(
            THINGSPEAK_URL,
            data=payload,
            headers={
                "Content-Type":
                "application/x-www-form-urlencoded"
            }
        )

        response_body = response.text.strip()

        print("HTTP status:", response.status_code)
        print("ThingSpeak response:", response_body)

        if response.status_code != 200:
            print("ThingSpeak HTTP error")
            return False

        if response_body == "0":
            print(
                "ThingSpeak rejected the update. "
                "Check API key or upload interval."
            )
            return False

        print(
            "Upload successful. Entry ID:",
            response_body
        )

        return True

    except Exception as error:
        print("ThingSpeak upload error:", error)
        return False

    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

        gc.collect()


# =========================================================
# LCD DRIVER + LAYOUT
# =========================================================

# Driver HD44780 qua bo mo rong I2C PCF8574, viet gon ngay trong file nay
# de khong phai upload them thu vien len board.
# Bit map cua bo mo rong: P0=RS, P1=RW, P2=EN, P3=den nen, P4..P7=D4..D7
_LCD_RS = 0x01
_LCD_EN = 0x04
_LCD_BACKLIGHT = 0x08

DEGREE = chr(0xDF)          # ky tu do trong bang ma A00 cua HD44780

DIRS_16 = (
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW"
)


class Lcd1602:
    def __init__(self, i2c, addr):
        self.i2c = i2c
        self.addr = addr
        self.buf = bytearray(1)
        self._setup()

    def _raw(self, value):
        self.buf[0] = value | _LCD_BACKLIGHT
        self.i2c.writeto(self.addr, self.buf)

    def _pulse(self, value):
        self._raw(value | _LCD_EN)
        sleep_us(1)
        self._raw(value & 0xFB)
        sleep_us(50)

    def _write4(self, value):
        self._raw(value)
        self._pulse(value)

    def _write(self, value, rs=0):
        self._write4((value & 0xF0) | rs)
        self._write4(((value << 4) & 0xF0) | rs)

    def _setup(self):
        sleep_ms(50)

        # Ba lan 0x30 de ep ve che do 8 bit da biet, roi chuyen sang 4 bit
        for _ in range(3):
            self._write4(0x30)
            sleep_ms(5)

        self._write4(0x20)
        sleep_ms(1)

        self._write(0x28)   # 4 bit, 2 dong, font 5x8
        self._write(0x08)   # tat hien thi
        self._write(0x01)   # xoa man hinh
        sleep_ms(2)
        self._write(0x06)   # con tro tu tang
        self._write(0x0C)   # bat hien thi, an con tro

    def line(self, row, text):
        self._write(0x80 | (0x40 if row else 0x00))

        # MicroPython KHÔNG có str.ljust() (chỉ CPython mới có), nên tự chèn
        # khoảng trắng. Phải ghi đủ 16 ký tự để xoá phần thừa của lần vẽ trước.
        text = text[:LCD_COLS]
        text = text + " " * (LCD_COLS - len(text))

        for ch in text:
            code = ord(ch)
            self._write(code if code < 256 else 0x3F, _LCD_RS)


_lcd = None
_lcd_failures = 0


def _open_i2c(sda, scl):
    """
    Mở I2C trên một cặp chân và quét bus.
    Trả về (bus, danh_sach_dia_chi). Ưu tiên I2C phần cứng, không được thì SoftI2C.
    """

    from machine import Pin, I2C

    try:
        bus = I2C(0, scl=Pin(scl), sda=Pin(sda), freq=100000)
    except Exception:
        from machine import SoftI2C
        bus = SoftI2C(scl=Pin(scl), sda=Pin(sda), freq=100000)

    return bus, bus.scan()


def lcd_init():
    """
    Khoi tao LCD. Moi loi deu bi nuot: man hinh chi la phu, khong duoc
    phep lam chet phan doc cam bien va upload.
    """

    global _lcd

    if not LCD_ENABLED:
        return

    # Thử đúng cặp chân khai báo trước; nếu không thấy thiết bị nào thì thử đảo
    # SDA/SCL rồi in ra cặp thực sự chạy được. Đảo hai dây này là nhầm lẫn rất
    # hay gặp, và bắt board tự trả lời thì chắc hơn là ngồi đoán.
    for sda, scl in (
        (LCD_SDA_PIN, LCD_SCL_PIN),
        (LCD_SCL_PIN, LCD_SDA_PIN)
    ):
        try:
            i2c, found = _open_i2c(sda, scl)
        except Exception as error:
            print(
                "LCD: khong mo duoc I2C tren SDA={}, SCL={}: {}".format(
                    sda, scl, error
                )
            )
            continue

        if not found:
            print(
                "LCD: khong thay thiet bi I2C nao tren SDA={}, SCL={}".format(
                    sda, scl
                )
            )
            continue

        if LCD_ADDR:
            addr = LCD_ADDR
        elif 0x27 in found:
            addr = 0x27
        elif 0x3F in found:
            addr = 0x3F
        else:
            addr = found[0]

        try:
            _lcd = Lcd1602(i2c, addr)
        except Exception as error:
            print("LCD: khoi tao man hinh that bai:", error)
            _lcd = None
            continue

        print(
            "LCD: san sang tai {} voi SDA={}, SCL={}{}".format(
                hex(addr),
                sda,
                scl,
                "" if sda == LCD_SDA_PIN else "  <-- DAO so voi khai bao!"
            )
        )
        return

    print("LCD: bo qua. Tram van chay binh thuong.")
    _lcd = None


def lcd_write(top, bottom):
    global _lcd, _lcd_failures

    if _lcd is None:
        return

    try:
        _lcd.line(0, top)
        _lcd.line(1, bottom)
        _lcd_failures = 0

    except Exception as error:
        _lcd_failures += 1
        print("LCD: loi ghi:", error)

        if _lcd_failures >= 3:
            print("LCD: tat han sau 3 loi lien tiep. Tram van chay binh thuong.")
            _lcd = None


def compass(deg):
    return DIRS_16[int((deg % 360) / 22.5 + 0.5) % 16]


def fmt_lux(value):
    """
    Anh sang 0..200000 lux gon trong toi da 4 ky tu.
    0 -> "0", 999 -> "999", 1234 -> "1.2k", 200000 -> "200k"
    """

    if value is None:
        return "--"

    if value < 1000:
        return str(value)

    # Cận trên là 9950 chứ không phải 10000: 9999/1000 làm tròn thành "10.0k",
    # dài 5 ký tự và phá mất bố cục.
    if value < 9950:
        return "{:.1f}k".format(value / 1000.0)

    return "{}k".format(int(round(value / 1000.0)))


def pad_row(left, right):
    """
    Hai cot: trai cang trai, phai cang phai, khoang trong don o giua.
    Chieu rong thay doi thi khoang giua co gian, nen bo cuc khong bao gio vo.
    """

    gap = LCD_COLS - len(left) - len(right)

    if gap < 1:
        # Lưới an toàn cuối cùng. Người gọi phải rút gọn ở mức TRƯỜNG dữ liệu
        # trước khi tới đây: cắt giữa một con số sẽ tạo ra giá trị sai mà vẫn
        # đọc được như thật (200000 lux -> "200"), tệ hơn là không hiện gì.
        return (left + " " + right)[:LCD_COLS]

    return left + (" " * gap) + right


def lcd_lines(temperature, pm2_5, light, wind_speed, wind_angle):
    """
    Dung hai dong 16 ky tu. Ham thuan tuy, khong dung toi phan cung,
    nen kiem thu duoc tren may tinh.

    Dong 1: toc do gio + huong chu + goc
    Dong 2: nhiet do + PM2.5 + anh sang
    Gia tri doc hong hien "--" chu khong giu lai so cu.
    """

    if wind_speed is None:
        left, right = "Wind", "-- ERR"
    elif wind_speed <= CALM_MS or wind_angle is None:
        left, right = "{:.1f}m/s".format(wind_speed), "CALM"
    else:
        left = "{:.1f}m/s {}".format(wind_speed, compass(wind_angle))
        right = "{:.0f}{}".format(wind_angle, DEGREE)

    top = pad_row(left, right)

    # Tiền tố L để số bên phải không bị đọc nhầm là một phần của PM
    lux_txt = "L" + fmt_lux(light)

    # Khi chật, rút gọn theo bậc ở mức TRƯỜNG dữ liệu: bỏ số lẻ nhiệt độ trước,
    # rồi rút nhãn PM. Không bao giờ cắt chuỗi - mất một chữ số thập phân hay
    # một chữ cái nhãn thì nhìn ra ngay, còn một con số bị cắt cụt vẫn đọc được
    # như số thật và sẽ đánh lừa người xem (200000 lux hiện thành "200").
    for temp_fmt, pm_prefix in (
        ("{:.1f}C", "PM"),
        ("{:.0f}C", "PM"),
        ("{:.0f}C", "P")
    ):
        temp_txt = "--" if temperature is None else temp_fmt.format(temperature)
        pm_txt = pm_prefix + ("--" if pm2_5 is None else str(pm2_5))
        left = temp_txt + " " + pm_txt

        if len(left) + 1 + len(lux_txt) <= LCD_COLS:
            break

    bottom = pad_row(left, lux_txt)

    return top, bottom


# =========================================================
# MAIN PROGRAM
# =========================================================

print()
print("--- System Start ---")
print("ESP32-WROOM Modbus Environmental Station")
print("ThingSpeak channel:", CHANNEL_ID)
print(
    "UART1 RX={}, TX={}, baud=4800".format(
        RX_PIN,
        TX_PIN
    )
)
print()

lcd_init()
lcd_write("VFCD station", "Connecting WiFi")

connect_wifi()

if wlan.isconnected():
    # Hiện IP một nhịp để tiện tra khi cần SSH/serial
    lcd_write("WiFi connected", wlan.ifconfig()[0])
    sleep_ms(1500)
else:
    lcd_write("WiFi FAILED", "check credentials")
    sleep_ms(1500)

# Đặt thời điểm upload trước đó lùi về quá khứ
# để lần đọc đầu tiên được gửi ngay.
last_upload = ticks_ms() - UPLOAD_INTERVAL_MS

# Lần upload đầu sau khi khởi động được đánh dấu BOOT., để dashboard biết chắc
# board vừa reset thay vì phải đoán qua các giá trị cảm biến bằng 0.
first_upload = True


while True:
    # Chỉ chứa dữ liệu đọc thành công trong chu kỳ hiện tại
    fields = {}

    # Tên các lệnh Modbus đọc hỏng trong chu kỳ này, theo đúng thứ tự đọc
    failed_reads = []
    attempted_reads = 4 if READ_WIND else 3

    # Đặt lại mỗi chu kỳ: LCD phải hiện "--" khi đọc hỏng,
    # không được giữ lại giá trị cũ của chu kỳ trước.
    temperature = humidity = noise = None
    pm2_5 = pm10 = pressure = None
    wind_speed = wind_direction = wind_angle = None

    print("----------------------------------------")
    print("Reading sensors...")

    # -----------------------------------------------------
    # Temperature, humidity and noise
    # -----------------------------------------------------

    thn = read_temperature_humidity_noise()

    if thn is not None:
        temperature, humidity, noise = thn

        print(
            "Temperature = {:.1f} C".format(
                temperature
            )
        )
        print(
            "Humidity    = {:.1f} %RH".format(
                humidity
            )
        )
        print(
            "Noise       = {:.1f} dB".format(
                noise
            )
        )

        fields["field3"] = temperature
        fields["field6"] = humidity
        fields["field7"] = noise

    else:
        failed_reads.append("THN")

    # -----------------------------------------------------
    # Light
    # -----------------------------------------------------

    lux = read_light()

    if lux is not None:
        print("Light       = {} lux".format(lux))
        fields["field5"] = lux

    else:
        failed_reads.append("LUX")

    # -----------------------------------------------------
    # PM2.5, PM10 and pressure
    # -----------------------------------------------------

    pm = read_pm_pressure()

    if pm is not None:
        pm2_5, pm10, pressure = pm

        print(
            "PM2.5       = {} ug/m3".format(
                pm2_5
            )
        )
        print(
            "PM10        = {} ug/m3".format(
                pm10
            )
        )
        print(
            "Pressure    = {:.1f} kPa".format(
                pressure
            )
        )

        fields["field4"] = pressure
        fields["field8"] = pm2_5

        # PM10 chưa được gửi vì channel hiện đã dùng đủ 8 field.

    else:
        failed_reads.append("PM")

    # -----------------------------------------------------
    # Wind
    # -----------------------------------------------------

    if READ_WIND:
        wind = read_wind()

        if wind is not None:
            wind_speed, wind_direction, wind_angle = wind

            print(
                "Wind Speed = {:.2f} m/s".format(
                    wind_speed
                )
            )
            print(
                "Wind Code  = {}".format(
                    wind_direction
                )
            )
            print(
                "Wind Angle = {} deg".format(
                    wind_angle
                )
            )

            fields["field1"] = wind_speed

            # Field 2 gửi góc hướng gió.
            fields["field2"] = wind_angle

            # Để gửi mã hướng gió thay vì góc, dùng:
            # fields["field2"] = wind_direction

        else:
            failed_reads.append("WIND")

    # -----------------------------------------------------
    # LCD
    # -----------------------------------------------------

    if len(failed_reads) >= attempted_reads:
        lcd_write("SENSOR NO REPLY", "Check RS485/pwr")
    else:
        top_line, bottom_line = lcd_lines(
            temperature,
            pm2_5,
            lux,
            wind_speed,
            wind_angle
        )
        lcd_write(top_line, bottom_line)

    # -----------------------------------------------------
    # ThingSpeak upload
    # -----------------------------------------------------

    current_time = ticks_ms()

    if (
        ticks_diff(current_time, last_upload)
        >= UPLOAD_INTERVAL_MS
    ):
        status = build_status(
            failed_reads,
            attempted_reads,
            first_upload
        )

        print()
        print("Status      = {}".format(status))

        if send_to_thingspeak(fields, status):
            first_upload = False

        # Ghi nhận cả lần gửi thành công hoặc thất bại,
        # tránh gửi liên tục gây vượt giới hạn ThingSpeak.
        last_upload = ticks_ms()

    print()
    sleep_ms(SENSOR_INTERVAL_MS)

