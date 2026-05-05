import java.util.*;
public class Java_Test2 {
    public static void main(String[] args) {
        Scanner s = new Scanner(System.in);
        String num = s.next();
        System.out.println("Number: " + num);
        for (int i = 0; i < 3; i++) {
            num = num * 2;
        }
        System.out.println("New Number: " + num);
    }
}