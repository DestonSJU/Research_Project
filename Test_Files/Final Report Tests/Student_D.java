public class Student_D {
    public static void main(String[] args) {
        double num = 4;
        System.out.println("New Number: " + doubleNumber(num));
    }
    public static double doubleNumber(double n) {
        n = n * 2;
        return Math.round(n);
    }
}
