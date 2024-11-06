"""from kafka import KafkaConsumer
consumer = KafkaConsumer('clovis', bootstrap_servers='51.38.185.58:9092')
for msg in consumer:
    print(msg)
"""

from aiokafka import AIOKafkaConsumer
import asyncio


async def receive():
    consumer = AIOKafkaConsumer(
        "processed",
        bootstrap_servers='51.38.185.58:9092'
    )

    await consumer.start()
    try:
        async for msg in consumer:
            print(
                "{}:{:d}:{:d}: key={} value={} timestamp_ms={}".format(
                    msg.topic, msg.partition, msg.offset, msg.key, msg.value,
                    msg.timestamp)
            )
    finally:
        await consumer.stop()


asyncio.run(receive())
