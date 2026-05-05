public class Student_I {
    public static void main(String[] args) {
        double number = 10;
        System.out.println("New Number: " + doubleNumber(number));
    }
    public static double doubleNumber(double number) {
        return Math.round(Math.abs(number));
    }
}
