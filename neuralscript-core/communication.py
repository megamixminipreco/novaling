# neuralscript-core/communication.py
import queue # Using queue for a simple shared message channel

class Message:
    def __init__(self, sender_id, content, sequence_num):
        self.sender_id = sender_id
        self.content = content # Could be a simple string or a more complex object
        self.sequence_num = sequence_num # Basic way to track messages

    def __repr__(self):
        return f"Message(sender='{self.sender_id}', content='{self.content}', seq={self.sequence_num})"

class CommunicationChannel:
    def __init__(self):
        # For MVP, a simple in-memory queue for each potential receiver.
        # A real system would use sockets, message brokers (like RabbitMQ/Kafka), etc.
        self.channels = {} # Key: receiver_id, Value: queue.Queue

    def _get_channel_for(self, receiver_id):
        if receiver_id not in self.channels:
            self.channels[receiver_id] = queue.Queue()
        return self.channels[receiver_id]

    def send(self, receiver_id, message: Message):
        print(f"Channel: Attempting to send message from {message.sender_id} to {receiver_id}: {message.content}")
        receiver_channel = self._get_channel_for(receiver_id)
        receiver_channel.put(message)
        print(f"Channel: Message queued for {receiver_id}")

    def receive(self, receiver_id, timeout=1):
        receiver_channel = self._get_channel_for(receiver_id)
        try:
            message = receiver_channel.get(timeout=timeout)
            print(f"Channel: {receiver_id} received message: {message.content}")
            return message
        except queue.Empty:
            print(f"Channel: No message for {receiver_id} within timeout.")
            return None

class AIInstance:
    def __init__(self, instance_id, channel: CommunicationChannel):
        self.instance_id = instance_id
        self.channel = channel
        self.message_sequence = 0

    def send_message(self, receiver_id, content):
        self.message_sequence += 1
        message = Message(
            sender_id=self.instance_id,
            content=content,
            sequence_num=self.message_sequence
        )
        print(f"{self.instance_id}: Sending message to {receiver_id}: '{content}'")
        self.channel.send(receiver_id, message)

    def receive_message(self):
        print(f"{self.instance_id}: Checking for messages...")
        message = self.channel.receive(self.instance_id)
        if message:
            print(f"{self.instance_id}: Got message from {message.sender_id}: '{message.content}' (Seq: {message.sequence_num})")
            # Basic echo for demonstration
            if "ping" in message.content.lower() and message.sender_id != self.instance_id:
                self.send_message(message.sender_id, f"Echo reply to ping from {self.instance_id}")
            return message
        else:
            print(f"{self.instance_id}: No new messages.")
            return None

if __name__ == '__main__':
    # 1. Setup the communication channel
    shared_channel = CommunicationChannel()

    # 2. Create two AI instances
    ai1 = AIInstance(instance_id="AI_Alice", channel=shared_channel)
    ai2 = AIInstance(instance_id="AI_Bob", channel=shared_channel)

    print("\n--- Simulating Basic Communication ---")
    
    # 3. AI Alice sends a message to AI Bob
    ai1.send_message(receiver_id="AI_Bob", content="Hello Bob! This is Alice. (Ping 1)")
    
    # 4. AI Bob tries to receive it
    ai2.receive_message() # Bob should receive Alice's message and send an echo

    # 5. AI Alice tries to receive Bob's echo
    ai1.receive_message()

    print("\n--- Simulating No Message ---")
    # 6. AI Bob checks for messages again (should be none)
    ai2.receive_message()

    print("\n--- Simulating Ping-Pong ---")
    # 7. Alice sends another ping
    ai1.send_message(receiver_id="AI_Bob", content="Are you there, Bob? (Ping 2)")
    # 8. Bob receives and echos
    received_by_bob = ai2.receive_message()
    # 9. Alice receives Bob's echo
    if received_by_bob: 
        ai1.receive_message()
    
    print("\n--- Test sending to self (should ideally be handled or disallowed based on design) ---")
    ai1.send_message(receiver_id="AI_Alice", content="Testing self message.")
    ai1.receive_message()
