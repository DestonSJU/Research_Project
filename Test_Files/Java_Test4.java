import java.lang.Math;
public class Java_Test4 {
    public static void main(String[] args) {
        String sNum = "10";
        System.out.println("New Number: " + doubleStringNumber(sNum));
    }
    public static int doubleStringNumber(String s) {
        int number = Integer.parseInt(s);
        number = number * 2;
        return Math.abs(number);
    }
}