# Contact Manager - Java OOP Learning Exercise

## Quick Start

Welcome to the Contact Manager exercise! This project will help you master **Object-Oriented Programming** concepts in Java including inheritance, polymorphism, abstract classes, and interfaces.

## What You'll Build

A contact management system that can:
- Store people's contact information (phone numbers and emails)
- Send messages to contacts
- Support different contact types:
  - **Landline Phone** - Audio messages only (voice calls)
  - **Mobile Phone** - Both audio AND text messages
  - **Email** - Text messages only
- Spam all contacts with a message (demonstrates polymorphism!)

## Project Structure

```
SE_Design_Ex2_dh320-master/
├── README.md                    ← You are here
├── LEARNING_GUIDE.md            ← Detailed tutorial (READ THIS FIRST!)
├── src/contacts/
│   ├── ✅ Person.java           (Complete - simple data class)
│   ├── ✅ Audio.java            (Complete - audio message wrapper)
│   ├── ✅ ContactInfo.java      (Complete - abstract base class)
│   ├── ✅ Contact.java          (Complete - wrapper class)
│   ├── ✅ AudioMessageEnabled   (Complete - interface)
│   ├── ✅ TextMessageEnabled    (Complete - interface)
│   ├── 🎯 Phone.java            (YOUR TASK - implement this)
│   ├── 🎯 MobilePhone.java      (YOUR TASK - implement this)
│   ├── 🎯 Email.java            (YOUR TASK - implement this)
│   └── 🎯 ContactManager.java   (YOUR TASK - implement this)
├── test/contacts/               (Test suites)
└── SOLUTION_REFERENCE/          (Original solutions - try not to peek!)
```

## Getting Started

### 1. Open in IntelliJ IDEA

```bash
# Navigate to this directory
cd Software-Engineering-Design/SE_Design_Ex2_dh320-master
```

Then in IntelliJ:
- **File → Open** → Select this folder
- Mark `src` as Sources Root (right-click → Mark Directory as → Sources Root)
- Mark `test` as Test Sources Root
- Add JUnit library (IntelliJ should prompt you)

### 2. Read the Learning Guide

**IMPORTANT**: Open `LEARNING_GUIDE.md` first! It contains:
- Detailed explanations of OOP concepts
- Class hierarchy diagrams
- Step-by-step instructions for each task
- Hints and examples
- Common pitfalls to avoid

### 3. Understand the Class Hierarchy

```
                    ContactInfo (abstract)
                           |
         +-----------------+-----------------+
         |                                   |
      Phone                                Email
         |                                   |
    implements:                         implements:
    AudioMessageEnabled                 TextMessageEnabled
         |
    MobilePhone
         |
    implements:
    AudioMessageEnabled + TextMessageEnabled
```

### 4. Implement the Classes

Work in this order:

#### Task 1: Phone.java ⭐⭐ (Start here!)
- Landline phone that sends audio messages
- Extends `ContactInfo`
- Implements `AudioMessageEnabled`
- **Learn**: Abstract classes, inheritance, interfaces

#### Task 2: MobilePhone.java ⭐⭐⭐
- Mobile phone that sends both text AND audio messages
- Extends `Phone` (inherits audio capability)
- Implements `TextMessageEnabled` (adds text capability)
- **Learn**: Class inheritance, method overriding, polymorphism

#### Task 3: Email.java ⭐⭐
- Email contact that sends text messages
- Extends `ContactInfo`
- Implements `TextMessageEnabled`
- **Learn**: Similar to Phone but different interface

#### Task 4: ContactManager.java ⭐⭐⭐⭐
- Manages collection of contacts
- Add contacts, retrieve by person, send to all
- **Learn**: Collections (ArrayList), polymorphism in action

### 5. Run Tests

Each class has a corresponding test file:

```bash
# Run all tests
Right-click on 'test/contacts' → Run 'All Tests'

# Run individual tests
PhoneTest.java         # Tests Task 1
EmailTest.java         # Tests Task 3
MobilePhoneTest.java   # Tests Task 2
ContactManagerTest.java # Tests Task 4
```

### 6. Test-Driven Development

Recommended workflow:
1. Read the task in `LEARNING_GUIDE.md`
2. Open the corresponding `.java` file (e.g., `Phone.java`)
3. Read the TODO comments and hints
4. Run the test (it will fail - that's expected!)
5. Implement the methods
6. Run the test again
7. Fix errors and iterate until test passes
8. Move to next task

## Key Concepts You'll Learn

### Abstract Classes
- `ContactInfo` is abstract - cannot be instantiated
- Defines common contract for all contact types
- Provides structure for subclasses

### Interfaces
- `AudioMessageEnabled` - defines ability to send audio messages
- `TextMessageEnabled` - defines ability to send text messages
- Classes can implement multiple interfaces

### Inheritance
- `Phone extends ContactInfo`
- `MobilePhone extends Phone`
- Inherits fields and methods from parent

### Polymorphism
The real magic! When you call `sendMessage()`:
- On a `Phone` → sends audio message
- On a `MobilePhone` → sends text message (overrides Phone!)
- On an `Email` → sends email message

The `ContactManager` doesn't need to know which type it is!

## Expected Output Formats

### Phone (Audio)
```
contacts.Audio saying 'Hello'+114 53 132
```

### MobilePhone (Text)
```
Hello+434 9434 132
```

### Email
```
Hello: emailP1@address.com
```

## Example Usage

```java
// Create contact manager
ContactManager cm = new ContactManager();

// Create a person
Person alice = new Person("Alice");

// Add contact information
cm.add(alice, new Phone("123-456"));
cm.add(alice, new MobilePhone("789-012"));
cm.add(alice, new Email("alice@email.com"));

// Get all contacts for Alice
List<ContactInfo> aliceContacts = cm.contactDetails(alice);
// Returns: [Phone, MobilePhone, Email]

// Send message to all contacts
cm.spam("Hello!");
// Output:
// contacts.Audio saying 'Hello!'+123-456
// Hello!+789-012
// Hello!: alice@email.com
```

## IntelliJ Tips

### Essential Shortcuts
- `Ctrl+O` - Override methods (very useful for this exercise!)
- `Ctrl+I` - Implement interface methods
- `Alt+Insert` - Generate code (constructor, getters, etc.)
- `Ctrl+Space` - Code completion
- `Ctrl+Click` - Jump to definition
- `Ctrl+H` - View class hierarchy
- `Ctrl+Alt+L` - Reformat code

### Debugging
- Click gutter to set breakpoint
- `Shift+F9` - Start debugging
- `F8` - Step over
- `F7` - Step into
- Inspect variables in debugger panel

## Common Mistakes

1. **Forgetting `@Override`** - Always use it when overriding methods
2. **Wrong output format** - Check the examples carefully!
3. **MobilePhone.sendMessage()** - Should send TEXT, not audio
4. **Using `equals()` instead of `==`** - Use `==` for Person comparison in `contactDetails()`
5. **Forgetting `super()`** - MobilePhone must call `super(phoneNumber)` in constructor
6. **Wrong field access** - Use `protected` so subclasses can access fields

## Need Help?

1. **Read LEARNING_GUIDE.md** - Contains detailed explanations
2. **Check the error message** - Java errors are usually helpful
3. **Look at the test** - Shows expected behavior
4. **Use IntelliJ's hints** - Red squiggles often have quick fixes (Alt+Enter)
5. **View class hierarchy** - Press Ctrl+H on ContactInfo to see structure
6. **Only if stuck** - Check `SOLUTION_REFERENCE/` folder

## What Makes This Exercise Great?

This is NOT just about completing tasks. You're learning:

- **How real-world systems are designed** - Using abstract classes and interfaces to define contracts
- **Code reuse** - MobilePhone inherits from Phone instead of duplicating code
- **Flexibility** - Easy to add new contact types (Fax, InstantMessage, etc.)
- **Polymorphism** - Write code once (spam), works for all types
- **Clean architecture** - Separation of concerns, clear responsibilities

## After Completing

Try these extensions:
1. Add a `Fax` class (audio only, like Phone)
2. Add filtering methods to ContactManager
3. Validate phone numbers and email addresses
4. Add a `Person` search method
5. Create a command-line interface
6. Add ability to group contacts

## Files You'll Modify

- ✏️ `src/contacts/Phone.java`
- ✏️ `src/contacts/MobilePhone.java`
- ✏️ `src/contacts/Email.java`
- ✏️ `src/contacts/ContactManager.java`

## Files That Are Complete (Don't Modify)

- ✅ `src/contacts/Person.java`
- ✅ `src/contacts/Audio.java`
- ✅ `src/contacts/ContactInfo.java`
- ✅ `src/contacts/Contact.java`
- ✅ `src/contacts/AudioMessageEnabled.java`
- ✅ `src/contacts/TextMessageEnabled.java`

---

**Ready to start?** Open `LEARNING_GUIDE.md` and begin with Task 1 (Phone.java)!

Remember: Programming is learned by doing. Don't just read - write the code! 🚀
