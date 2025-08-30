import random
from collections import deque

# Queue (FIFO)
strawberry_queue = deque()

# Stack (LIFO)
borderline_stack = []

pick_count = 0
unpick_count = 0
borderline_count = 0

# Add 5 strawberries for quick test
for i in range(1, 6):
    strawberry_queue.append(f"Strawberry-{i}")

# Process queue
while strawberry_queue:
    strawberry = strawberry_queue.popleft()
    decision = random.choice([0, 1, 2])  # 0=Unpick, 1=Pick, 2=Borderline
    
    if decision == 1:
        print(f"{strawberry} → Pick ✅")
        pick_count += 1
    elif decision == 0:
        print(f"{strawberry} → Unpick ❌")
        unpick_count += 1
    else:
        print(f"{strawberry} → Borderline ⚠️ (sent to stack)")
        borderline_stack.append(strawberry)
        borderline_count += 1

# Re-check borderline
print("\n🔄 Re-checking borderline strawberries...")
while borderline_stack:
    strawberry = borderline_stack.pop()
    decision = random.choice([0, 1])  # re-check only pick/unpick
    if decision == 1:
        print(f"{strawberry} (re-check) → Pick ✅")
        pick_count += 1
    else:
        print(f"{strawberry} (re-check) → Unpick ❌")
        unpick_count += 1

# Final Report
print("\n--- 📊 Summary Report ---")
print(f"Total Pick: {pick_count}")
print(f"Total Unpick: {unpick_count}")
print(f"Borderline (Re-checked): {borderline_count}")
