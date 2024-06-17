echo "Formatting code..."
find {src,test} -name "*.java" -exec java -jar lib/google-java-format-1.15.0-all-deps.jar -r {} \;
echo "Done."
