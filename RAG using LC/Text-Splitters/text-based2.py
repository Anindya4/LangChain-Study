# this is same as text-based splitting but for special files like code block or markdown

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,    # any language can be split by chagning the language type
    chunk_size = 550,
    chunk_overlap = 0
)

code = """
class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.available = True

    def __str__(self):
        status = "Available" if self.available else "Checked out"
        return f"{self.title} by {self.author} ({self.isbn}) - {status}"

    def borrow(self):
        if self.available:
            self.available = False
            return True
        return False

    def return_book(self):
        self.available = True


class Member:
    def __init__(self, name, member_id):
        self.name = name
        self.member_id = member_id
        self.borrowed_books = []

    def borrow_book(self, book):
        if book.borrow():
            self.borrowed_books.append(book)
            print(f"{self.name} borrowed '{book.title}'.")
        else:
            print(f"Sorry, '{book.title}' is not available.")

    def return_book(self, book):
        if book in self.borrowed_books:
            book.return_book()
            self.borrowed_books.remove(book)
            print(f"{self.name} returned '{book.title}'.")
        else:
            print(f"{self.name} doesn't have '{book.title}' borrowed.")


class Library:
    def __init__(self):
        self.books = []
        self.members = []

    def add_book(self, book):
        self.books.append(book)

    def add_member(self, member):
        self.members.append(member)

    def list_books(self):
        print("\nLibrary Catalog:")
        for book in self.books:
            print(" -", book)


# Example usage
if __name__ == "__main__":
    library = Library()
    book1 = Book("1984", "George Orwell", "9780451524935")
    book2 = Book("To Kill a Mockingbird", "Harper Lee", "9780060935467")

    member = Member("Alice", "M001")

    library.add_book(book1)
    library.add_book(book2)
    library.add_member(member)

    library.list_books()
    member.borrow_book(book1)
    member.borrow_book(book1)
    member.return_book(book1)
    library.list_books()

"""

result = splitter.split_text(code)
print('-------------------------------------------------------------------------')
print(result[1])
print('-------------------------------------------------------------------------')
print(len(result))
