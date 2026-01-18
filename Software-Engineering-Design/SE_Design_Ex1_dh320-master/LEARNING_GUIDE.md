# Picture Processor - Java Learning Exercise

## Overview
This exercise will help you practice:
- Java fundamentals (loops, conditionals, methods)
- Object-oriented programming (classes, objects, encapsulation)
- Working with 2D arrays and pixel manipulation
- Command-line argument parsing
- IntelliJ IDEA features

## Project Structure
```
SE_Design_Ex1_dh320-master/
├── src/picture/
│   ├── Color.java              (COMPLETE - utility class for RGB colors)
│   ├── Picture.java            (COMPLETE - wrapper for BufferedImage)
│   └── PictureProcessor.java   (YOUR TASK - implement image operations)
├── test/picture/
│   └── PictureProcessorTest.java (Test suite to validate your work)
└── images/                      (Test images for processing)
```

## Your Mission
Implement the image processing methods in `PictureProcessor.java`. The Picture and Color classes are already complete and provide these useful methods:

### Picture API
- `Picture(String filepath)` - Load an image from file
- `Picture(int width, int height)` - Create a blank image
- `int getWidth()` / `int getHeight()` - Get dimensions
- `Color getPixel(int x, int y)` - Get color at position
- `void setPixel(int x, int y, Color color)` - Set color at position
- `void saveAs(String filepath)` - Save image to file

### Color API
- `Color(int red, int green, int blue)` - Create color (0-255 range)
- `int red()` / `int green()` / `int blue()` - Get color components

## Tasks (Implement in this order)

### Task 1: Grayscale Conversion ⭐ (Easy)
**Method**: `public static void grayscale(String in, String out)`

**Goal**: Convert a color image to grayscale (monochrome).

**Algorithm**:
1. Load the input image
2. Create a new output image with the same dimensions
3. For each pixel:
   - Get the RGB values
   - Calculate average: `(red + green + blue) / 3`
   - Create a new color with average value for all three components
   - Set the pixel in output image
4. Save the output image

**Test**: Run `grayscaleBlack` and `grayscaleRainbow` tests

**IntelliJ Tips**:
- Use `Ctrl+Space` for code completion
- Use `Ctrl+Alt+V` to extract variables
- Use `fori` live template for loops

---

### Task 2: Invert Colors ⭐ (Easy)
**Method**: `public static void invert(String in, String out)`

**Goal**: Invert the colors of an image (like a photographic negative).

**Algorithm**:
1. Load the input image
2. Create output image
3. For each pixel:
   - Get RGB values
   - Invert: `new_value = 255 - old_value`
   - Create new color with inverted values
   - Set pixel in output
4. Save output

**Think About**: Why do we use `255 - value` for inversion?

**Test**: Run `invertBlack` and `invertRainbow` tests

---

### Task 3: Rotate Image ⭐⭐ (Medium)
**Method**: `public static void rotate(String angle, String in, String out)`

**Goal**: Rotate an image by 90, 180, or 270 degrees.

**Algorithm Hints**:

**For 180° rotation**:
- Output dimensions = Input dimensions
- Pixel at `(x, y)` goes to `(width-x-1, height-y-1)`

**For 90° clockwise rotation**:
- Output width = Input height, Output height = Input width
- Pixel at `(x, y)` goes to `(height-y-1, x)`

**For 270° clockwise rotation** (90° counter-clockwise):
- Output width = Input height, Output height = Input width
- Pixel at `(x, y)` goes to `(y, width-x-1)`

**Implementation Tip**: Use an if-else chain to check the angle parameter.

**Test**: Run `rotate90Green`, `rotate90BlueRect`, `rotate180BlueRect`, `rotate270BlueRect` tests

**Challenge**: Draw on paper what happens to corner pixels during rotation!

---

### Task 4: Flip Image ⭐⭐ (Medium)
**Method**: `public static void flip(String dir, String in, String out)`

**Goal**: Flip an image horizontally (H) or vertically (V).

**Algorithm**:

**Horizontal flip**:
- Pixel at `(x, y)` goes to `(width-x-1, y)`

**Vertical flip**:
- Pixel at `(x, y)` goes to `(x, height-y-1)`

**Implementation**: Use a conditional to check if `dir.equals("H")` or `"V"`

**Test**: Run `flipVGreen`, `flipVBlue`, `flipHBlue` tests

---

### Task 5: Blur Effect ⭐⭐⭐ (Hard)
**Method**: `public static void blur(String in, String out)`

**Goal**: Apply a blur effect by averaging each pixel with its 8 neighbors.

**Algorithm**:
1. For each pixel at position `(x, y)`:
2. Look at the 3x3 grid of pixels centered on `(x, y)`:
   ```
   (x-1,y-1) (x,y-1) (x+1,y-1)
   (x-1,y)   (x,y)   (x+1,y)
   (x-1,y+1) (x,y+1) (x+1,y+1)
   ```
3. Average the RGB values of all valid neighbors
4. Handle edge cases: pixels on borders don't have all 8 neighbors

**Helper Method Suggestion**: Create `neighbour_average(int x, int y, Picture in, Picture out)`

**Edge Case Handling**:
- If pixel is on the border, either:
  - Keep original value (easier)
  - Or only average available neighbors (more accurate)

**Test**: Run `blurBWPatterns` and `blurSunset` tests

**IntelliJ Tip**: Use `Ctrl+Alt+M` to extract the neighbor averaging logic into a method

---

### Task 6: Blend Images ⭐⭐⭐ (Hard)
**Method**: `public static void blend(List<String> variables)`

**Goal**: Blend multiple images together by averaging their pixel values.

**Parameters**:
- `variables` is a list where:
  - All elements except the last are input image paths
  - The last element is the output path

**Algorithm**:
1. Find the minimum width and height among all input images
2. Create output image with those dimensions
3. For each pixel position:
   - Sum the RGB values from all input images
   - Divide by the number of images
   - Set the averaged color in output
4. Save to the last path in variables list

**Challenges**:
- Working with List instead of arrays
- Handling variable number of inputs
- Finding minimum dimensions

**Test**: Run `blendBWAndRainbow` and `blendRainbowSunset` tests

---

### Task 7: Mosaic Pattern ⭐⭐⭐⭐ (Advanced)
**Method**: `public static void mosaic(List<String> variables)`

**Goal**: Create a mosaic by alternating between images in a tile pattern.

**Parameters**:
- `variables[0]` = tile size
- `variables[1..n-1]` = input image paths
- `variables[n]` = output path

**Algorithm** (This is challenging!):
1. Parse tile size from first element
2. Find minimum dimensions among input images
3. For each pixel `(x, y)`:
   - Determine which tile it belongs to: `tileX = x / tileSize`
   - Select image based on tile position (alternate in a pattern)
   - Get pixel from that image
4. The pattern should rotate as you move across rows

**This is an advanced challenge** - try to understand the solution pattern!

---

### Task 8: Command-Line Interface ⭐⭐ (Medium)
**Method**: `public static void main(String[] args)`

**Goal**: Parse command-line arguments and call the appropriate method.

**Commands to support**:
```
help                                     - display help menu
grayscale <in> <out>                     - convert to grayscale
rotate 90|180|270 <in> <out>             - rotate image
invert <in> <out>                        - invert colors
flip H|V <in> <out>                      - flip horizontally/vertically
blur <in> <out>                          - blur image
blend <in_1> <in_2> ... <out>            - blend multiple images
mosaic <tile-size> <in_1> <in_2> ... <out> - create mosaic
```

**Implementation Steps**:
1. Check if args is empty → show help message
2. Check if args[0] equals "help" → show full help menu
3. For each command, verify:
   - Correct number of arguments
   - Valid parameters (angles, directions)
   - Call the corresponding method
4. Show error message for invalid input

**IntelliJ Tips**:
- Use `sout` live template for `System.out.println()`
- Use `Ctrl+Alt+L` to reformat code

---

## Running Tests in IntelliJ

### Setup:
1. Open the project in IntelliJ
2. Right-click on `test/picture/PictureProcessorTest.java`
3. Select "Run 'PictureProcessorTest'"

### Running Individual Tests:
- Click the green arrow next to any `@Test` method
- Or right-click the method and select "Run"

### Debugging:
- Set breakpoints by clicking in the gutter (left margin)
- Use `Shift+F9` to start debugging
- Use `F8` to step over, `F7` to step into

### Test-Driven Development Approach:
1. Pick a task (e.g., grayscale)
2. Run the related tests - they will fail
3. Implement the method
4. Run tests again - iterate until they pass
5. Move to next task

---

## IntelliJ Features to Practice

### Code Generation:
- `Alt+Insert` - Generate code (constructors, getters, etc.)
- `Ctrl+O` - Override methods
- `Ctrl+I` - Implement methods

### Refactoring:
- `Ctrl+Alt+V` - Extract variable
- `Ctrl+Alt+M` - Extract method
- `Shift+F6` - Rename
- `Ctrl+Alt+L` - Reformat code

### Navigation:
- `Ctrl+Click` - Go to definition
- `Ctrl+B` - Go to declaration
- `Alt+F7` - Find usages
- `Ctrl+E` - Recent files

### Code Completion:
- `Ctrl+Space` - Basic completion
- `Ctrl+Shift+Space` - Smart completion
- Live templates: `fori`, `sout`, `psvm`, etc.

---

## Learning Objectives

By completing this exercise, you will:
- ✅ Practice nested loops (essential for 2D image processing)
- ✅ Understand coordinate systems and transformations
- ✅ Work with objects and method calls
- ✅ Handle command-line arguments
- ✅ Use conditional logic (if-else chains)
- ✅ Work with Lists and arrays
- ✅ Perform mathematical calculations
- ✅ Use IntelliJ's testing framework
- ✅ Practice debugging techniques
- ✅ Write clean, readable code

---

## Recommended Order of Implementation

**Day 1**: Tasks 1-2 (Grayscale, Invert) - Build confidence
**Day 2**: Tasks 3-4 (Rotate, Flip) - Practice coordinate transformations
**Day 3**: Task 5 (Blur) - Challenge yourself with neighbor logic
**Day 4**: Task 6 (Blend) - Work with multiple inputs
**Day 5**: Task 8 (CLI) - Put it all together
**Optional**: Task 7 (Mosaic) - Advanced challenge

---

## Common Pitfalls to Avoid

1. **Off-by-one errors**: Remember arrays are 0-indexed, but width/height are 1-indexed
2. **X/Y confusion**: x is column (width), y is row (height)
3. **Modifying input**: Always create a new Picture for output
4. **Not handling edge cases**: Test with different image sizes
5. **Forgetting to save**: Don't forget `pictureOut.saveAs(out)`

---

## Extension Ideas (After completing main tasks)

1. Implement additional filters:
   - Brightness adjustment
   - Contrast adjustment
   - Sepia tone effect
   - Edge detection

2. Add validation:
   - Check if files exist before processing
   - Validate color values are in 0-255 range
   - Handle invalid arguments gracefully

3. Optimize performance:
   - Reduce redundant Picture object creation in blend/mosaic
   - Use parallel processing for large images

---

## Getting Help

When stuck:
1. Read the error message carefully
2. Check the test to understand expected behavior
3. Use IntelliJ's debugger to inspect variables
4. Draw the transformation on paper
5. Compare your output images with expected results

Good luck! Remember: programming is learned by doing. Don't just read the guide - write the code! 🚀
