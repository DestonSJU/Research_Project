public class Student_F {
    public static int doubleNumber(double number) {
        number = number * 2;
        return Math.abs(number);
    }
    public static void main(String[] args) {
        double number = 10.4;
        System.out.println("New Number: " + doubleNumber(number));
    }
}
