package com.lg.graph;

/**
 * Created by lg on 2017/12/11.
 */

import java.io.BufferedInputStream;
        import java.io.File;
        import java.io.IOException;
        import java.io.InputStream;
        import java.net.URL;
        import java.net.Socket;
// import java.net.HttpURLConnection;
        import java.net.URLConnection;
        import java.util.ArrayList;
        import java.util.InputMismatchException;
        import java.util.Locale;
        import java.util.NoSuchElementException;
        import java.util.Scanner;
        import java.util.regex.Pattern;
public final class In {
    private static final String CHARSET_NAME = "UTF-8";
    private static final Locale LOCALE = Locale.US;
    private static final Pattern WHITESPACE_PATTERN
            = Pattern.compile("\\p{javaWhitespace}+");
    private static final Pattern EMPTY_PATTERN
            = Pattern.compile("");
    private static final Pattern EVERYTHING_PATTERN
            = Pattern.compile("\\A");
    private Scanner scanner;

    //System.in通过缓冲流包装了一下
    //然后设定语言环境
    public In() {
        scanner = new Scanner(new BufferedInputStream(System.in), CHARSET_NAME);
        scanner.useLocale(LOCALE);
    }

    //从套接字接受输入流
    public In(Socket socket) {
        if (socket == null) throw new NullPointerException("argument is null");
        try {
            InputStream is = socket.getInputStream();
            scanner = new Scanner(new BufferedInputStream(is), CHARSET_NAME);
            scanner.useLocale(LOCALE);
        } catch (IOException ioe) {
            throw new IllegalArgumentException("Could not open " + socket);
        }
    }

    //从url接受输入流
    public In(URL url) {
        if (url == null) throw new NullPointerException("argument is null");
        try {
            URLConnection site = url.openConnection();
            InputStream is = site.getInputStream();
            scanner = new Scanner(new BufferedInputStream(is), CHARSET_NAME);
            scanner.useLocale(LOCALE);
        } catch (IOException ioe) {
            throw new IllegalArgumentException("Could not open " + url);
        }
    }

    //从文件里面接受输入流
    public In(File file) {
        if (file == null) throw new NullPointerException("argument is null");
        try {
            scanner = new Scanner(file, CHARSET_NAME);
            scanner.useLocale(LOCALE);
        } catch (IOException ioe) {
            throw new IllegalArgumentException("Could not open " + file);
        }
    }

    //从文件名或者网站名接受输入流，先试读文件，再试读网站
    public In(String name) {
        if (name == null) throw new NullPointerException("argument is null");
        try {
            // first try to read file from local file system
            File file = new File(name);
            if (file.exists()) {
                scanner = new Scanner(file, CHARSET_NAME);
                scanner.useLocale(LOCALE);
                return;
            }
            // next try for files included in jar
            URL url = getClass().getResource(name);
            // or URL from web
            if (url == null) {
                url = new URL(name);
            }
            URLConnection site = url.openConnection();
            // in order to set User-Agent, replace above line with these two
            // HttpURLConnection site = (HttpURLConnection) url.openConnection();
            // site.addRequestProperty("User-Agent", "Mozilla/4.76");
            InputStream is = site.getInputStream();
            scanner = new Scanner(new BufferedInputStream(is), CHARSET_NAME);
            scanner.useLocale(LOCALE);
        } catch (IOException ioe) {
            throw new IllegalArgumentException("Could not open " + name);
        }
    }

    public In(Scanner scanner) {
        if (scanner == null) throw new NullPointerException("argument is null");
        this.scanner = scanner;
    }

    //一些和Scanner类相关的方法
    public boolean exists() {
        return scanner != null;
    }

    public boolean isEmpty() {
        return !scanner.hasNext();
    }

    public boolean hasNextLine() {
        return scanner.hasNextLine();
    }

    //判断是否还有字符，包括whitespace
    public boolean hasNextChar() {
        scanner.useDelimiter(EMPTY_PATTERN);//换成空字符串分割
        boolean result = scanner.hasNext();
        scanner.useDelimiter(WHITESPACE_PATTERN);//再换回java默认的
        return result;
    }

    public String readLine() {
        String line;
        try {
            line = scanner.nextLine();
        } catch (NoSuchElementException e) {
            line = null;
        }
        return line;
    }

    public char readChar() {
        scanner.useDelimiter(EMPTY_PATTERN);
        String ch = scanner.next();
        assert ch.length() == 1 : "Internal (Std)In.readChar() error!"
                + " Please contact the authors.";//这里使用了断言
        scanner.useDelimiter(WHITESPACE_PATTERN);
        return ch.charAt(0);
    }

    public String readAll() {
        if (!scanner.hasNextLine())
            return "";
        String result = scanner.useDelimiter(EVERYTHING_PATTERN).next();//以字符串开头
        // not that important to reset delimeter, since now scanner is empty
        scanner.useDelimiter(WHITESPACE_PATTERN); // but let's do it anyway
        return result;
    }

    public String readString() {
        return scanner.next();
    }

    public int readInt() {
        return scanner.nextInt();
    }

    public double readDouble() {
        return scanner.nextDouble();
    }

    public float readFloat() {
        return scanner.nextFloat();
    }

    public long readLong() {
        return scanner.nextLong();
    }

    public short readShort() {
        return scanner.nextShort();
    }

    public byte readByte() {
        return scanner.nextByte();
    }

    //读布尔值
    public boolean readBoolean() {
        String s = readString();
        if (s.equalsIgnoreCase("true")) return true;
        if (s.equalsIgnoreCase("false")) return false;
        if (s.equals("1")) return true;
        if (s.equals("0")) return false;
        throw new InputMismatchException();
    }

    public String[] readAllStrings() {
        // we could use readAll.trim().split(), but that's not consistent
        // since trim() uses characters 0x00..0x20 as whitespace
        String[] tokens = WHITESPACE_PATTERN.split(readAll());
        if (tokens.length == 0 || tokens[0].length() > 0)
            return tokens;
        String[] decapitokens = new String[tokens.length - 1];//如果第一个字符串长度为0，就切除
        for (int i = 0; i < tokens.length - 1; i++)
            decapitokens[i] = tokens[i + 1];
        return decapitokens;
    }

    public String[] readAllLines() {
        ArrayList<String> lines = new ArrayList<String>();
        while (hasNextLine()) {
            lines.add(readLine());
        }
        return lines.toArray(new String[0]);
    }

    public int[] readAllInts() {
        String[] fields = readAllStrings();
        int[] vals = new int[fields.length];
        for (int i = 0; i < fields.length; i++)
            vals[i] = Integer.parseInt(fields[i]);
        return vals;
    }

    public double[] readAllDoubles() {
        String[] fields = readAllStrings();
        double[] vals = new double[fields.length];
        for (int i = 0; i < fields.length; i++)
            vals[i] = Double.parseDouble(fields[i]);
        return vals;
    }

    public void close() {
        scanner.close();
    }
}