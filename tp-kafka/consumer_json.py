import numpy as np
from aiokafka import AIOKafkaConsumer
import asyncio
import json


async def receive():
    consumer = AIOKafkaConsumer(
        "clovisfbv",
        bootstrap_servers='51.38.185.58:9092'
    )

    await consumer.start()
    try:
        async for msg in consumer:
            try:
                dico = json.loads(msg.value.decode("UTF-8"))
                sum_d = np.sum(dico['data'])
                print(sum_d)
            except:
                print("Hanne Segard, en marche arrière")
    finally:
        await consumer.stop()


asyncio.run(receive())
