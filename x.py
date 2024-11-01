import secrets
import string

def generate_random_id(length=8):
    alphabet = string.ascii_letters + string.digits
    random_id = ''.join(secrets.choice(alphabet) for _ in range(length))
    return random_id

# Generate a random ID
random_id = generate_random_id()
print("Random ID:", random_id)
