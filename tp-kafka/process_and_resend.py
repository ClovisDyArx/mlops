import numpy as np
import time
import aiokafka
import asyncio
import json


def serializer(value):
    return json.dumps(value).encode("UTF-8")


async def receive_and_send():
    consumer = aiokafka.AIOKafkaConsumer(
        "clovisfbv",
        bootstrap_servers='51.38.185.58:9092'
    )

    producer = aiokafka.AIOKafkaProducer(
        bootstrap_servers="51.38.185.58:9092"
    )

    await consumer.start()
    try:
        async for msg in consumer:
            try:
                dico = json.loads(msg.value.decode("UTF-8"))
                sum_d = np.sum(dico['data'])
                result = f"Clovis : {sum_d}"
                print(result)
                await producer.send_and_wait("processed", serializer(result))
            except:
                print("Hanne Segard, en marche arrière")
    finally:
        await consumer.stop()


asyncio.run(receive_and_send())
