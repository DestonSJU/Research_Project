import java.lang.Math;
public class Student_A {
    public static void main(String[] args) {
        double num = 10;
        System.out.println("New Number: " + doubleNumber(num));
    }
    public static double doubleNumber(double n) {
        n = n * 2;
        return Math.round(Math.abs(n));
    }
}