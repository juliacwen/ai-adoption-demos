import csv
import random

# Stripe official test cards (good ones)
STRIPE_TEST_CARDS = [
    "4242424242424242",  # Visa
    "4000056655665556",  # Visa debit
    "5555555555554444",  # Mastercard
    "2223003122003222",  # Mastercard (2-series)
    "378282246310005",   # Amex
    "6011111111111117",  # Discover
    "30569309025904",    # Diners Club
    "3566002020360505",  # JCB
]

def luhn_checksum(card_number: str) -> int:
    """Calculate Luhn checksum for card validation."""
    def digits_of(n):
        return [int(d) for d in str(n)]
    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d*2))
    return checksum % 10

def generate_bad_card():
    """Generate a random invalid card number."""
    length = random.choice([12, 13, 14, 15, 16, 19])
    card = "".join(str(random.randint(0, 9)) for _ in range(length))
    # Ensure it fails Luhn
    if luhn_checksum(card) == 0:
        # flip last digit to break it
        card = card[:-1] + str((int(card[-1]) + 5) % 10)
    return card

def generate_dataset(filename="training_data.csv", n_rows=200):
    fieldnames = ["email", "amount", "payment_method", "card_number", "label"]

    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n_rows):
            email = f"user{i}@example.com"
            amount = random.choice([25, 30, 40, 50, 60, 70, 80, 100])
            payment_method = "Card"

            if random.random() < 0.5:
                # Good case: Stripe test card
                card = random.choice(STRIPE_TEST_CARDS)
                label = "good"
            else:
                # Bad case: random invalid card
                card = generate_bad_card()
                label = "bad"

            writer.writerow({
                "email": email,
                "amount": amount,
                "payment_method": payment_method,
                "card_number": card,
                "label": label
            })

    print(f"✅ Generated {n_rows} rows in {filename}")

if __name__ == "__main__":
    generate_dataset("training_data.csv", n_rows=500)

