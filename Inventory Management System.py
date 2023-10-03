import json


class InventoryItem:
    def __init__(self, item_id, name, quantity, price):
        self.item_id = item_id
        self.name = name
        self.quantity = quantity
        self.price = price


class InventorySystem:
    def __init__(self):
        self.inventory = []

    def add_item(self, item):
        self.inventory.append(item)

    def find_item(self, item_id):
        for item in self.inventory:
            if item.item_id == item_id:
                return item
        return None

    def update_item(self, item_id, new_quantity, new_price):
        item = self.find_item(item_id)
        if item:
            item.quantity = new_quantity
            item.price = new_price
            return True
        return False

    def delete_item(self, item_id):
        item = self.find_item(item_id)
        if item:
            self.inventory.remove(item)
            return True
        return False

    def display_inventory(self):
        print("\nInventory\n")
        for item in self.inventory:
            print(f"ID: {item.item_id}, Name: {item.name}, Quantity: {item.quantity}, Price: {item.price}")

    def save_inventory(self, filename):
        with open(filename, 'w') as file:
            serialized_data = [item.__dict__ for item in self.inventory]
            json.dump(serialized_data, file)

    def load_inventory(self, filename):
        with open(filename, 'r') as file:
            serialized_data = json.load(file)
            self.inventory = [InventoryItem(**item_data) for item_data in serialized_data]


def get_input_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid Input. Please Enter a Valid Integer.")


def get_input_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid Input. Please Enter a Valid Number")


if __name__ == "__main__":
    system = InventorySystem()

    try:
        system.load_inventory("Inventory_Data.json")
    except FileNotFoundError:
        print("No such directory found. The file does not Exist.")

    while True:
        print("\n1.) Add Item")
        print("2.) Update Item")
        print("3.) Delete Item")
        print("4.) Display Inventory")
        print("5.) Save and Exit")

        choice = get_input_int("\nEnter your Choice: ")

        if choice == 1:
            item_id = get_input_int("\nEnter Item ID: ")
            name = input("Enter Item Name: ")
            quantity = get_input_int("Enter Item Quantity: ")
            price = get_input_float("Enter Item Price: ")

            new_item = InventoryItem(item_id, name, quantity, price)
            system.add_item(new_item)
            print("Item Added Successfully.")

        elif choice == 2:
            item_id = get_input_int("\nEnter Item ID to Update: ")
            new_quantity = get_input_int("Enter New Quantity: ")
            new_price = get_input_float("Enter New Price: ")

            if system.update_item(item_id, new_quantity, new_price):
                print("Item Updated Successfully.")
            else:
                print("Item Not Found.")

        elif choice == 3:
            item_id = get_input_int("\nEnter Item ID to Delete: ")
            if system.delete_item(item_id):
                print("Item Deleted Successfully.")
            else:
                print("Item Not Found.")

        elif choice == 4:
            if system.inventory:
                system.display_inventory()
            else:
                print("\nInventory is Empty. Please Choose Option 1 to add Items.")
                print("\nThank You!!!")

        elif choice == 5:
            system.save_inventory("Inventory_Data.json")
            print("\nInventory Data Saved.")
            print("\nThank You!!!")
            break

        else:
            print("\nInvalid Choice. Please Select a Valid Option.")
