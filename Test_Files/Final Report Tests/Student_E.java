import java.lang.Math;
public class Student_E {
    public static void main(String[] args) {
        double number = 10;
        System.out.println("New Number: " + doubleNumber(number));
    }
    public static double doubleNumber(double number) {
        number = number * 2;
        return Math.round(Math.abs(number));
    }
}
