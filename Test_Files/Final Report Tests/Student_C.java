public class Student_C {
    public static void main(String[] args) {
        double num = 8.1;
        System.out.println("New Number: " + doubleNumber(num));
    }
    public static double doubleNumber(double n) {
        n = n * 2;
        return Math.abs(n);
    }
}

