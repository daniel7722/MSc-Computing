# Picture Processor - Java Learning Exercise

## Quick Start

Welcome to the Picture Processor exercise! This project will help you practice Java fundamentals and IntelliJ IDEA while building a real image processing application.

## What You'll Build

A command-line tool that can:
- Convert images to grayscale
- Rotate images (90°, 180°, 270°)
- Invert colors
- Flip images horizontally or vertically
- Apply blur effects
- Blend multiple images together
- Create mosaic patterns

## Getting Started

### 1. Open in IntelliJ IDEA

```bash
# If you're not already there, navigate to this directory
cd Software-Engineering-Design/SE_Design_Ex1_dh320-master
```

Then in IntelliJ:
- File → Open → Select this folder
- Mark `src` as Sources Root (right-click → Mark Directory as → Sources Root)
- Mark `test` as Test Sources Root
- Add JUnit library (IntelliJ should prompt you, or use Alt+Enter on red underlines)

### 2. Read the Learning Guide

Open **`LEARNING_GUIDE.md`** - this contains:
- Detailed explanations of each task
- Step-by-step algorithms
- Helpful hints and tips
- IntelliJ keyboard shortcuts
- Common pitfalls to avoid

### 3. Start Implementing

Open **`src/picture/PictureProcessor.java`** and start with Task 1 (grayscale).

The file contains:
- Method signatures with helpful comments
- Step-by-step TODO hints
- Javadoc explaining what each method should do

### 4. Test Your Code

Run the tests to verify your implementation:
- Open `test/picture/PictureProcessorTest.java`
- Click the green arrow next to any test method
- Or right-click and select "Run 'PictureProcessorTest'"

### 5. Use Test-Driven Development

Recommended workflow:
1. Read the task in LEARNING_GUIDE.md
2. Look at the method signature and hints in PictureProcessor.java
3. Run the related test (it will fail - that's expected!)
4. Implement the method
5. Run the test again
6. Fix any errors and iterate until the test passes
7. Move to the next task

## Available Files

| File | Purpose |
|------|---------|
| `LEARNING_GUIDE.md` | Comprehensive tutorial with detailed instructions |
| `README.md` | This file - quick start guide |
| `src/picture/PictureProcessor.java` | **YOUR TASK** - implement the methods here |
| `src/picture/Picture.java` | Complete - wrapper for image operations |
| `src/picture/Color.java` | Complete - RGB color representation |
| `test/picture/PictureProcessorTest.java` | Test suite to validate your work |
| `SOLUTION_REFERENCE.java` | Reference solution (try not to peek!) |
| `images/` | Sample images for testing |

## Recommended Task Order

1. **Task 1: Grayscale** (Easy) - Build confidence with basic pixel iteration
2. **Task 2: Invert** (Easy) - Practice color manipulation
3. **Task 3: Rotate** (Medium) - Learn coordinate transformations
4. **Task 4: Flip** (Medium) - More coordinate practice
5. **Task 8: CLI** (Medium) - Put it all together with command-line interface
6. **Task 5: Blur** (Hard) - Challenge yourself with neighbor logic
7. **Task 6: Blend** (Hard) - Work with multiple inputs
8. **Task 7: Mosaic** (Advanced) - Optional challenge

## Running Your Program

Once you've implemented the methods, you can run the program from the command line:

```bash
# Compile
javac -d bin src/picture/*.java

# Run examples
java -cp bin picture.PictureProcessor help
java -cp bin picture.PictureProcessor grayscale images/rainbow64x64doc.png output.png
java -cp bin picture.PictureProcessor rotate 90 images/green64x64doc.png output.png
```

Or use IntelliJ's Run configuration:
- Right-click on `PictureProcessor.java`
- Run 'PictureProcessor.main()'
- Edit configuration to add command-line arguments

## Need Help?

1. **Read the error message** - Java's error messages are usually helpful
2. **Check LEARNING_GUIDE.md** - detailed hints for each task
3. **Use the debugger** - set breakpoints and inspect variables
4. **Look at the tests** - they show expected behavior
5. **Draw it out** - sketch transformations on paper
6. **Only if stuck** - peek at `SOLUTION_REFERENCE.java`

## IntelliJ Tips

- `Ctrl+Space` - Code completion
- `Ctrl+Shift+F10` - Run current file
- `Shift+F10` - Rerun last configuration
- `Ctrl+Shift+F9` - Run tests in current file
- `Ctrl+Click` - Jump to definition
- `Alt+Enter` - Quick fix suggestions
- `Ctrl+Alt+L` - Reformat code

## Learning Goals

By completing this exercise, you'll practice:
- Nested loops and 2D iteration
- Object-oriented programming
- Method design and implementation
- Command-line argument parsing
- Test-driven development
- IntelliJ IDEA features
- Debugging techniques

## Have Fun!

Image processing is a great way to see your code produce visual results. Don't just implement the methods - experiment! Try the operations on different images, chain operations together, and understand what each transformation does visually.

Remember: **The best way to learn programming is by doing!** 🚀
