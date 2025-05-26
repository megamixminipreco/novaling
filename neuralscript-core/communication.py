# neuralscript-core/communication.py
import queue
import numpy as np # For message_vector and attention_pattern potentially

class Message:
    def __init__(self, sender_id, message_vector, attention_pattern, confidence_level, sequence_num=0):
        self.sender_id = sender_id
        self.message_vector = message_vector # e.g., np.array or list of floats
        self.attention_pattern = attention_pattern # e.g., np.array, list, or descriptive string
        self.confidence_level = confidence_level # float
        self.sequence_num = sequence_num # Optional, but kept for now

    def __repr__(self):
        return (f"Message(sender='{self.sender_id}', vec_len={len(self.message_vector) if self.message_vector is not None else 'N/A'}, "
                f"att_pat_type={type(self.attention_pattern).__name__}, conf={self.confidence_level:.2f}, seq={self.sequence_num})")

class CommunicationChannel:
    def __init__(self):
        self.channels = {} 

    def _get_channel_for(self, receiver_id):
        if receiver_id not in self.channels:
            self.channels[receiver_id] = queue.Queue()
        return self.channels[receiver_id]

    def send(self, receiver_id, message: Message):
        print(f"Channel: Attempting to send message from {message.sender_id} to {receiver_id} (Seq: {message.sequence_num})")
        receiver_channel = self._get_channel_for(receiver_id)
        receiver_channel.put(message)
        print(f"Channel: Message queued for {receiver_id}")

    def receive(self, receiver_id, timeout=1):
        receiver_channel = self._get_channel_for(receiver_id)
        try:
            message = receiver_channel.get(timeout=timeout)
            print(f"Channel: {receiver_id} received message: {message}")
            return message
        except queue.Empty:
            print(f"Channel: No message for {receiver_id} within timeout.")
            return None

class AIInstance:
    def __init__(self, instance_id, channel: CommunicationChannel):
        self.instance_id = instance_id
        self.channel = channel
        self.message_sequence_sent = 0

    def send_structured_message(self, receiver_id, message_vector, attention_pattern, confidence):
        self.message_sequence_sent += 1
        message = Message(
            sender_id=self.instance_id,
            message_vector=message_vector,
            attention_pattern=attention_pattern,
            confidence_level=confidence,
            sequence_num=self.message_sequence_sent
        )
        print(f"{self.instance_id}: Sending structured message to {receiver_id}: {message}")
        self.channel.send(receiver_id, message)

    def process_received_message(self):
        print(f"{self.instance_id}: Checking for messages...")
        message = self.channel.receive(self.instance_id)
        if message:
            print(f"{self.instance_id}: Got Message Details:")
            print(f"  Sender ID: {message.sender_id}")
            print(f"  Message Vector (type): {type(message.message_vector).__name__}, Length: {len(message.message_vector) if hasattr(message.message_vector, '__len__') else 'N/A'}")
            # print(f"  Message Vector (data): {message.message_vector}") # Could be verbose
            print(f"  Attention Pattern (type): {type(message.attention_pattern).__name__}")
            # print(f"  Attention Pattern (data): {message.attention_pattern}") # Could be verbose
            print(f"  Confidence: {message.confidence_level:.2f}")
            print(f"  Sequence Num: {message.sequence_num}")
            
            # Conceptual processing:
            # For example, decode_vector(message.message_vector) -> decoded_concepts
            # attend_to relevant_concepts in working_memory using message.attention_pattern
            # update_beliefs(decoded_concepts, message.confidence_level)
            print(f"{self.instance_id}: Conceptually processing message from {message.sender_id}...")
            
            # Simplified echo/response for testing channel still works
            if message.sender_id != self.instance_id and self.instance_id == "AI_Bob": # Bob echos
                 self.send_structured_message(
                     receiver_id=message.sender_id,
                     message_vector=np.array([0.9, 0.8, 0.7]), # Dummy echo vector
                     attention_pattern="echo_response_pattern",
                     confidence=0.95
                 )
            return message
        else:
            print(f"{self.instance_id}: No new messages.")
            return None

if __name__ == '__main__':
    shared_channel = CommunicationChannel()
    ai_alice = AIInstance(instance_id="AI_Alice", channel=shared_channel)
    ai_bob = AIInstance(instance_id="AI_Bob", channel=shared_channel)

    print("\n--- Simulating Structured Message Communication ---")
    
    # Alice sends a structured message to Bob
    alice_msg_vec = np.array([0.1, 0.2, 0.3, 0.4])
    alice_att_pat = {"type": "focus_area", "details": "concepts_A_B"}
    alice_conf = 0.85
    
    ai_alice.send_structured_message(
        receiver_id="AI_Bob",
        message_vector=alice_msg_vec,
        attention_pattern=alice_att_pat,
        confidence=alice_conf
    )
    
    # Bob processes the message (and might send an echo)
    ai_bob.process_received_message()
    
    # Alice checks for Bob's echo
    ai_alice.process_received_message()

    print("\n--- Simulating No Message for Alice ---")
    ai_alice.process_received_message()
