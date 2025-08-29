from collections import deque
import sys

# Initialize your global data structures
conveyor_belt = deque()
uncertain_stack = []
classification_report = {
    "Pick": 0,
    "Unpick": 0
}

def enqueue_strawberry(image_path):
    """Adds a strawberry image path to the queue."""
    global conveyor_belt
    conveyor_belt.append(image_path)
    print(f"Enqueued: {image_path}")

def dequeue_strawberry():
    """Removes and returns the first strawberry image path from the queue."""
    global conveyor_belt
    if conveyor_belt:
        return conveyor_belt.popleft()
    else:
        print("Conveyor belt is empty.")
        return None

def push_uncertain(image_path):
    """Adds an uncertain strawberry image path to the stack."""
    global uncertain_stack
    uncertain_stack.append(image_path)
    print(f"Pushed to uncertain stack: {image_path}")

def pop_uncertain():
    """Removes and returns the last uncertain strawberry image from the stack."""
    global uncertain_stack
    if uncertain_stack:
        return uncertain_stack.pop()
    else:
        print("Uncertain stack is empty.")
        return None

def update_report(classification):
    """Updates the count for a given classification."""
    global classification_report
    if classification in classification_report:
        classification_report[classification] += 1
        print(f"Updated report: {classification} count is now {classification_report[classification]}")
    else:
        print(f"Invalid classification: {classification}")

def simulate_ml_model(image_path):
    """
    A placeholder function for your Machine Learning model.
    Replace this with your actual model's prediction logic.
    It should return 'Pick' or 'Unpick'.
    """
    print(f"Processing image: {image_path}")
    # For demonstration, we'll use a simple rule.
    if "good" in image_path.lower() or "pickable" in image_path.lower():
        return "Pick"
    elif "doubtful" in image_path.lower():
        return "Uncertain"
    else:
        return "Unpick"
    
# --- Main Project Flow ---
if __name__ == "__main__":
    # The script will now ask you for the image path
    image_to_check = input("Please insert the image path: ")
    
    # Check if the user provided an empty path
    if not image_to_check:
        print("No image path provided. Exiting.")
        sys.exit()

    print(f"\n--- Starting processing for image: {image_to_check} ---\n")
    
    # Enqueue and process the single image from the command line
    enqueue_strawberry(image_to_check)
    
    current_strawberry = dequeue_strawberry()
    if current_strawberry:
        classification = simulate_ml_model(current_strawberry)
        
        if classification == "Uncertain":
            push_uncertain(current_strawberry)
            print("Classification uncertain. Moved to stack for re-checking.\n")
        else:
            update_report(classification)
            print(f"Final classification for {current_strawberry}: {classification}\n")

    print("--- Processing complete ---\n")
    
    # Print the summary report and handle uncertain cases
    print("Final Summary Report:")
    for status, count in classification_report.items():
        print(f"Total {status} strawberries: {count}")
    
    print("\n--- Re-checking Uncertain Strawberries ---\n")
    while uncertain_stack:
        recheck_strawberry = pop_uncertain()
        print(f"Re-checking {recheck_strawberry}...")
        final_decision = simulate_ml_model(recheck_strawberry)
        if final_decision != "Uncertain":
            update_report(final_decision)
            print(f"Final decision after re-check: {final_decision}\n")
        else:
            print(f"Still uncertain. {recheck_strawberry} will be discarded.\n")