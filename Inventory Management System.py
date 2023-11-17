import json
from typing import List, Optional
import sqlite3

connect = sqlite3.connect('inventory.db')

# Create a connection string
conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=<your_server_name>;'
    r'DATABASE=<your_database_name>;'
    r'Trusted_Connection=yes;'
)

cursor = connect.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        price REAL NOT NULL
    )
""")

connect.commit()

cursor.execute("SELECT * FROM items")
rows = cursor.fetchall()
for row in rows:
    print(row)

connect.close()


class InventoryException(Exception):
    pass


class ItemNotFoundException(InventoryException):
    pass


class InventoryItem:
    def __init__(self, item_id: int, name: str, quantity: int, price: float, reorder_level: int, classification: str):
        self.item_id = item_id
        self.name = name
        self.quantity = quantity
        self.price = price
        self.reorder_level = reorder_level
        self.classification = classification


def log_action(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        print(f"Action Logged: {func.__name__}")
        return result

    return wrapper


class InventorySystem:
    def __init__(self):
        self.inventory: List[InventoryItem] = []

    @log_action
    def add_item(self, item: InventoryItem):
        self.inventory.append(item)

    def find_item(self, item_id: int) -> Optional[InventoryItem]:
        for item in self.inventory:
            if item.item_id == item_id:
                return item
        return None

    @log_action
    def update_item(self, item_id: int, new_quantity: int, new_price: float):
        item = self.find_item(item_id)
        if item:
            item.quantity = new_quantity
            item.price = new_price
            if item.quantity < item.reorder_level:
                self.create_order(item)
        else:
            raise ItemNotFoundException(f"Item with ID {item_id} not found.")

    @staticmethod
    def create_order(item: InventoryItem):
        print(f"Creating new order for item {item.item_id}")
        # Add code here to create a new order

    @log_action
    def delete_item(self, item_id: int):
        item = self.find_item(item_id)
        if item:
            self.inventory.remove(item)
        else:
            raise ItemNotFoundException(f"Item with ID {item_id} not found.")

    def display_inventory(self):
        print("\nInventory\n")
        for item in self.inventory:
            print(f"ID: {item.item_id}, Name: {item.name}, Quantity: {item.quantity}, Price: {item.price}")

    def save_inventory(self, filename: str):
        with open(filename, 'w') as file:
            serialized_data = [item.__dict__ for item in self.inventory]
            json.dump(serialized_data, file)

    def load_inventory(self, filename: str):
        with open(filename, 'r') as file:
            serialized_data = json.load(file)
            self.inventory = [InventoryItem(**item_data) for item_data in serialized_data]


if __name__ == "__main__":
    system = InventorySystem()

    while True:
        print("\n1.) Add Item")
        print("2.) Update Item")
        print("3.) Delete Item")
        print("4.) Display Inventory")
        print("5.) Save and Exit")

        choice = int(input("\nEnter your Choice: "))

        if choice == 1:
            item_id = int(input('Enter item ID: '))
            name = input('Enter item name: ')
            quantity = int(input('Enter quantity: '))
            price = float(input('Enter price: '))
            reorder_level = input('Enter Reorder Level: ')
            classification = input('Enter classification: ')
            item = InventoryItem(item_id, name, quantity, price, reorder_level, classification)
            system.add_item(item)

        elif choice == 2:
            item_id = int(input('Enter item ID to Update: '))
            new_quantity = int(input('Enter new quantity: '))
            new_price = float(input('Enter new price: '))
            try:
                system.update_item(item_id, new_quantity, new_price)
            except ItemNotFoundException:
                print(f"Item with ID {item_id} not found.")

        elif choice == 3:
            item_id = int(input('Enter item ID to Delete: '))
            try:
                system.delete_item(item_id)
            except ItemNotFoundException:
                print(f"Item with ID {item_id} not found.")

        elif choice == 4:
            system.display_inventory()

        elif choice == 5:
            filename = input('Enter filename to save inventory: ')
            system.save_inventory(filename)
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
