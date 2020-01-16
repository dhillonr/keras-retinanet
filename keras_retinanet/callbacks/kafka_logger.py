import keras
from kafka import KafkaProducer

class Kafka_Logger(keras.callbacks.Callback):

    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

    def on_epoch_end(self, epoch, logs={}):
        logs['epoch'] = epoch+1
        self.producer.send('training', str(logs).encode())