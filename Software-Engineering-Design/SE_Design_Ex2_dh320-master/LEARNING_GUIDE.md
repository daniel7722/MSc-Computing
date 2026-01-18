# Contact Manager - Java OOP Learning Exercise

## Overview
This exercise will help you practice:
- **Object-Oriented Programming** (inheritance, polymorphism, abstraction)
- **Abstract classes** vs **Interfaces**
- **Class hierarchies** and extending classes
- **Multiple interface implementation**
- **Collections** (ArrayList, filtering)
- **Method overriding** and the `@Override` annotation

## Project Structure
```
SE_Design_Ex2_dh320-master/
├── src/contacts/
│   ├── Person.java              ✅ COMPLETE - simple data class
│   ├── Audio.java               ✅ COMPLETE - audio message wrapper
│   ├── ContactInfo.java         ✅ COMPLETE - abstract base class
│   ├── Contact.java             ✅ COMPLETE - person + contact info wrapper
│   ├── AudioMessageEnabled.java ✅ COMPLETE - interface for audio messaging
│   ├── TextMessageEnabled.java  ✅ COMPLETE - interface for text messaging
│   ├── Phone.java               🎯 YOUR TASK - landline phone implementation
│   ├── MobilePhone.java         🎯 YOUR TASK - mobile phone (extends Phone)
│   ├── Email.java               🎯 YOUR TASK - email implementation
│   └── ContactManager.java      🎯 YOUR TASK - manages all contacts
└── test/contacts/
    └── *Test.java               (Test suites to validate your work)
```

## Understanding the Class Hierarchy

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

## Provided Classes (Already Complete)

### 1. Person
Simple class representing a person with a name.
```java
Person john = new Person("John Doe");
String name = john.name();  // "John Doe"
```

### 2. Audio
Wrapper for audio messages.
```java
Audio msg = new Audio("Hello");
System.out.println(msg);  // "contacts.Audio saying 'Hello'"
```

### 3. ContactInfo (Abstract Class)
Defines the contract for all contact information types.
```java
public abstract class ContactInfo {
  public abstract String contactInfo();      // Get the contact detail (phone/email)
  public abstract String contactInfoType();  // Get the type ("phone" or "email")
  public abstract void sendMessage(String msg);  // Send a message
}
```

### 4. Contact
Links a Person with their ContactInfo.
```java
Contact c = new Contact(person, phoneNumber);
Person p = c.getP();       // Get the person
ContactInfo info = c.getC(); // Get the contact info
```

### 5. AudioMessageEnabled (Interface)
```java
public interface AudioMessageEnabled {
  void sendAudioMessage(Audio msg);
}
```

### 6. TextMessageEnabled (Interface)
```java
public interface TextMessageEnabled {
  void sendTextMessage(String msg);
}
```

---

## Your Tasks

### Task 1: Implement the Phone Class ⭐⭐

**Goal**: Create a landline phone that can only send audio messages (voice calls).

**Class Definition**:
```java
public class Phone extends ContactInfo implements AudioMessageEnabled
```

**Requirements**:
1. **Field**: `protected String phoneNumber` - stores the phone number
2. **Constructor**: `Phone(String phoneNumber)` - initialize the phone number
3. **Override** `contactInfo()` - return the phone number
4. **Override** `contactInfoType()` - return `"phone"`
5. **Override** `sendMessage(String msg)` - convert to Audio and send
6. **Implement** `sendAudioMessage(Audio msg)` - print audio message format

**Output Format for sendAudioMessage**:
```
contacts.Audio saying 'Hello'+114 53 132
```
(Audio message toString + phone number, no spaces)

**Hints**:
- Use `System.out.println()` to send messages
- The `Audio` class already has a `toString()` method
- `sendMessage()` should create an `Audio` object and call `sendAudioMessage()`

**Why protected?**: The `phoneNumber` field is `protected` so that `MobilePhone` (which extends `Phone`) can access it.

---

### Task 2: Implement the MobilePhone Class ⭐⭐⭐

**Goal**: Create a mobile phone that can send BOTH text messages AND audio messages (inherits from Phone).

**Class Definition**:
```java
public class MobilePhone extends Phone implements AudioMessageEnabled, TextMessageEnabled
```

**Requirements**:
1. **Constructor**: `MobilePhone(String phoneNumber)` - call super constructor
2. **Override** `sendMessage(String msg)` - should send TEXT message (not audio!)
3. **Override** `sendAudioMessage(Audio msg)` - delegate to parent's implementation
4. **Implement** `sendTextMessage(String msg)` - print text message format

**Output Format for sendTextMessage**:
```
Hello+434 9434 132
```
(Message + phone number, no spaces or colons)

**Key Concept - Polymorphism**:
- `MobilePhone` IS-A `Phone` (inheritance)
- `MobilePhone` CAN-DO audio messaging (from `Phone`)
- `MobilePhone` CAN-DO text messaging (new capability)
- **Important**: When you call `sendMessage()` on a MobilePhone, it should send a TEXT message (overrides Phone's behavior)

**Hints**:
- Use `super(phoneNumber)` in constructor to call Phone's constructor
- Use `super.sendAudioMessage(msg)` to reuse Phone's audio implementation
- The behavior of `sendMessage()` is DIFFERENT from Phone!

---

### Task 3: Implement the Email Class ⭐⭐

**Goal**: Create an email contact that can send text messages.

**Class Definition**:
```java
public class Email extends ContactInfo implements TextMessageEnabled
```

**Requirements**:
1. **Field**: `protected String address` - stores the email address
2. **Constructor**: `Email(String address)` - initialize the email address
3. **Override** `contactInfo()` - return the email address
4. **Override** `contactInfoType()` - return `"email"`
5. **Override** `sendMessage(String msg)` - should call `sendTextMessage()`
6. **Implement** `sendTextMessage(String msg)` - print email message format

**Output Format for sendTextMessage** (and sendMessage):
```
Hello: emailP1@address.com
```
(Message + colon + space + email address)

**Hints**:
- Very similar structure to `Phone` class
- Both `sendMessage()` and `sendTextMessage()` should produce the same output
- Notice the format difference: Email uses `": "` but Phone doesn't

---

### Task 4: Implement the ContactManager Class ⭐⭐⭐⭐

**Goal**: Manage a collection of contacts and perform operations on them.

**Requirements**:

#### 4a. Fields and Constructor
```java
private final List<Contact> contacts = new ArrayList<>();
public ContactManager() {}  // Default constructor (no-op)
```

#### 4b. Method: `add(Person p, ContactInfo c)`
Add a new contact linking a person with their contact info.

**Algorithm**:
1. Create a new `Contact` object from the person and contact info
2. Add it to the `contacts` list

**Example**:
```java
ContactManager cm = new ContactManager();
cm.add(new Person("Alice"), new Email("alice@email.com"));
cm.add(new Person("Alice"), new MobilePhone("123-456"));
```

#### 4c. Method: `contactDetails(Person p)`
Returns a list of all ContactInfo for a given person.

**Return Type**: `List<ContactInfo>`

**Algorithm**:
1. Create an empty ArrayList to store results
2. Loop through all contacts
3. For each contact, check if the person matches (use `==` comparison)
4. If it matches, add the ContactInfo to the results list
5. Return the results list

**Example**:
```java
Person alice = new Person("Alice");
cm.add(alice, new Email("alice@email.com"));
cm.add(alice, new MobilePhone("123-456"));

List<ContactInfo> aliceContacts = cm.contactDetails(alice);
// Returns list with 2 items: Email and MobilePhone
```

**Hints**:
- Use a for-each loop: `for (Contact c : contacts)`
- Use `c.getP()` to get the Person from a Contact
- Use `c.getC()` to get the ContactInfo from a Contact
- Use `==` to compare Person objects (reference equality)

#### 4d. Method: `spam(String msg)`
Send a message to ALL contacts in the system.

**Algorithm**:
1. Loop through all contacts
2. For each contact, get the ContactInfo
3. Call `sendMessage(msg)` on that ContactInfo

**Polymorphism in Action**:
- Each ContactInfo type (Phone/MobilePhone/Email) implements `sendMessage()` differently
- Phone → sends Audio message
- MobilePhone → sends Text message
- Email → sends Email message
- The ContactManager doesn't need to know which type it is!

**Example Output** (if contacts contain Phone, Email, MobilePhone):
```
contacts.Audio saying 'Hello'+114 53 132
Hello: emailP1@address.com
Hello+434 9434 132
```

**Hints**:
- Very simple method - just loop and call
- Polymorphism does all the work for you!

---

## Running Tests in IntelliJ

### Setup:
1. Open the project in IntelliJ
2. Right-click on `test/contacts` folder
3. Select "Run 'All Tests'"

### Test Files:
- `PhoneTest.java` - Tests Phone class (Task 1)
- `MobilePhoneTest.java` - Tests MobilePhone class (Task 2)
- `EmailTest.java` - Tests Email class (Task 3)
- `ContactManagerTest.java` - Tests ContactManager class (Task 4)

### Running Individual Tests:
- Click the green arrow next to any `@Test` method
- Or right-click the test class and select "Run"

---

## Key OOP Concepts in This Exercise

### 1. Abstract Classes vs Interfaces

**Abstract Class (ContactInfo)**:
- Cannot be instantiated directly
- Can have both abstract and concrete methods
- Provides a template for subclasses
- Used for "IS-A" relationships

**Interface (AudioMessageEnabled, TextMessageEnabled)**:
- Pure contract - only method signatures
- A class can implement multiple interfaces
- Used for "CAN-DO" relationships
- Adds capabilities to classes

### 2. Inheritance Hierarchy

```
Phone IS-A ContactInfo
MobilePhone IS-A Phone (and therefore also IS-A ContactInfo)
Email IS-A ContactInfo
```

### 3. Polymorphism

When you have a `ContactInfo` reference, it could be pointing to:
- A `Phone` object
- A `MobilePhone` object
- An `Email` object

The correct `sendMessage()` implementation is called based on the ACTUAL object type, not the reference type:

```java
ContactInfo c1 = new Phone("123");
ContactInfo c2 = new Email("a@b.com");

c1.sendMessage("Hi");  // Calls Phone's sendMessage (audio)
c2.sendMessage("Hi");  // Calls Email's sendMessage (text)
```

### 4. Method Overriding

- Use `@Override` annotation for clarity and error checking
- Subclass can change behavior of inherited methods
- `super.method()` calls the parent's version

### 5. Multiple Interface Implementation

`MobilePhone` implements both `AudioMessageEnabled` and `TextMessageEnabled`:
```java
MobilePhone phone = new MobilePhone("123");
phone.sendTextMessage("Hi");   // From TextMessageEnabled
phone.sendAudioMessage(...);   // From AudioMessageEnabled
```

---

## Recommended Implementation Order

**Day 1**: Task 1 (Phone) - Understand abstract classes and basic inheritance
**Day 2**: Task 3 (Email) - Practice similar pattern with different interface
**Day 3**: Task 2 (MobilePhone) - Learn about extending classes and multiple interfaces
**Day 4**: Task 4 (ContactManager) - Put it all together with collections and polymorphism

---

## Common Pitfalls

1. **Forgetting `@Override`**: Always use it when overriding methods - helps catch errors
2. **Wrong output format**:
   - Phone/MobilePhone text: `msg + phoneNumber` (no spaces)
   - Phone audio: `audio.toString() + phoneNumber` (no spaces)
   - Email: `msg + ": " + address` (colon and space)
3. **MobilePhone.sendMessage()**: Should send TEXT, not audio (different from Phone!)
4. **Using `equals()` instead of `==`**: In `contactDetails()`, use `==` to compare Person objects
5. **Forgetting `super()`**: MobilePhone constructor must call `super(phoneNumber)`

---

## IntelliJ Tips

### Code Generation
- `Ctrl+O` - Override methods (very useful!)
- `Alt+Insert` - Generate constructor, getters, etc.
- Type `@Override` and press Enter - IntelliJ will suggest methods to override

### Refactoring
- `Ctrl+Alt+V` - Extract variable
- `Shift+F6` - Rename
- `Ctrl+Alt+L` - Reformat code

### Navigation
- `Ctrl+Click` - Jump to definition
- `Ctrl+H` - View type hierarchy (great for understanding inheritance!)
- `Alt+F7` - Find usages

---

## Extension Ideas (After Completing Main Tasks)

1. **Add more contact types**:
   - Fax (audio only, like Phone)
   - InstantMessage (text only, like Email)
   - VideoPhone (audio + text + video)

2. **Add filtering to ContactManager**:
   - `List<ContactInfo> getEmailContacts(Person p)`
   - `List<ContactInfo> getPhoneContacts(Person p)`
   - `void spamByType(String msg, String type)`

3. **Add validation**:
   - Validate phone number format
   - Validate email address format
   - Prevent duplicate contacts

4. **Add search functionality**:
   - Find person by name
   - Find contacts by phone number
   - Find contacts by email domain

---

## Understanding the Tests

### PhoneTest
Tests that Phone sends audio messages with correct format:
```java
Phone p1 = new Phone("+501 323 33");
p1.sendMessage("Hello");
// Expected output: "contacts.Audio saying 'Hello'+501 323 33"
```

### MobilePhoneTest
Tests that MobilePhone sends TEXT messages (not audio) and format is correct:
```java
MobilePhone p1 = new MobilePhone("+501 323 33");
p1.sendMessage("Hello");
// Expected output: "Hello+501 323 33" (NO "Audio" in output!)
```

### EmailTest
Tests email message format:
```java
Email email = new Email("myEmail@domain.com");
email.sendMessage("Hello");
// Expected output: "Hello: myEmail@domain.com"
```

### ContactManagerTest

**Test 1 - listsContactInfo**:
- Adds multiple contacts for same person
- Checks `contactDetails()` returns all of them
- Checks filtering works correctly per person

**Test 2 - spamContacts**:
- Adds various contact types
- Calls `spam("Hello")`
- Checks that all contacts received the message
- Each contact type formats it differently (polymorphism!)

---

## Learning Goals

By completing this exercise, you will:
- ✅ Understand the difference between abstract classes and interfaces
- ✅ Practice inheritance and extending classes
- ✅ Implement polymorphism in action
- ✅ Work with multiple interface implementation
- ✅ Use collections (ArrayList) to manage objects
- ✅ Filter and search through object collections
- ✅ Override methods correctly
- ✅ Understand the `protected` access modifier
- ✅ See how polymorphism enables flexible code design

---

## Getting Help

When stuck:
1. Read the error message - Java tells you what's wrong
2. Check the test to understand expected behavior
3. Review the class hierarchy diagram
4. Look at the provided classes (Person, Audio, ContactInfo) as examples
5. Use IntelliJ's "Implement Methods" feature (`Ctrl+I`)
6. Only if really stuck - peek at `SOLUTION_REFERENCE/` folder

Good luck! This exercise teaches fundamental OOP concepts you'll use throughout your Java career! 🚀
