import time
import aiokafka
import asyncio
import json


def serializer(value):
    return json.dumps(value).encode("UTF-8")


async def produce():
    producer = aiokafka.AIOKafkaProducer(bootstrap_servers="51.38.185.58:9092")
    await producer.start()
    try:
        for _ in range(10):
            time.sleep(1)
            message = {
                'data': [[1, 2], [3, 4]]
            }
            await producer.send_and_wait("clovisfbv", serializer(message))
    finally:
        await producer.stop()

asyncio.run(produce())
