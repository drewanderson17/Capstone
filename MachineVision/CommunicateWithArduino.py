import serial

serial = serial.Serial('/dev/ttyUSB0')
ser.write("test")
ser.close()