import java.io.File;
public class Java_Test3 {
    public static void main(String[] args) {
        File test_file = new File("Text1.txt");
        System.out.println("File Found: " + test_file.exists());
    }
}