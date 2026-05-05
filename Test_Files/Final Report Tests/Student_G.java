public class Student_G {
    public static void main(String[] args) {
        double number = 7.1;
        System.out.println("New Number: " + doubleNumber(number));
    }
    public static double doubleNumber(double number) {
        number = number * 2;
        return Math.ceil(Math.abs(number));
    }
}
