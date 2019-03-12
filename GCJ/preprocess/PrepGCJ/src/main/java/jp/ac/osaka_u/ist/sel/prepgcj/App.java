package jp.ac.osaka_u.ist.sel.prepgcj;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Hello world!
 *
 */
public class App {
	private static void print(String text) {
		System.out.println(text);
	}

	public static void main(String[] args) {
		// String text = new JavaMethodExtractor(Paths.get(args[0])).extract();
		// print(text);
		if (args.length != 2) {
			System.err.println("prep [src_dir] [dest_dir]");
			System.exit(1);
		}
		Path src = Paths.get(args[0]);
		Path dst = Paths.get(args[1]);
		List<Path> files = null;
		try {
			files = Files.walk(src).filter(Files::isRegularFile)
					.filter(file -> file.toString().toLowerCase().endsWith(".java")).collect(Collectors.toList());
		} catch (IOException e) {
			e.printStackTrace();
		}

		int count = 0;
		for (Path file : files) {
			String dirname = file.getParent().getFileName().toString();
			Path filename = Paths.get(dst.toString(), dirname, file.getFileName().toString());
			if (Files.exists(filename))
				continue;
			System.out.print(file + "...");
			String str = null;
			try {
				str = new JavaMethodExtractor(Paths.get(file.toString())).extract();
			} catch (Exception e) {
				System.err.println(file + "skip");
				continue;
			}
			try {
				Files.createDirectories(Paths.get(dst.toString(), dirname));
			} catch (IOException e) {
				e.printStackTrace();
			}
			try (BufferedWriter writer = Files.newBufferedWriter(filename)) {
				writer.write(str);
			} catch (IOException e) {
				e.printStackTrace();
			}
			count++;
			if (count > 10000) {
				System.gc();
				count = 0;
			}
			System.out.println(" done");

		}
		// List<String> texts = new JavaAnalyzer(args[0]).analyze();
		// System.out.println(CAnalyzer.analyze(args[0]));
		//String[] arr = { "cmd", "slack-post", "cpp_finish" };
		//execute(arr);
		System.out.println("finihed!");
	}

	public static String execute(String args[]) {
		String text = null;
		try {
			Process process = new ProcessBuilder(args).start();
			InputStream is = process.getInputStream();

			InputStreamReader isr = new InputStreamReader(is);

			BufferedReader reader = new BufferedReader(isr);
			StringBuilder builder = new StringBuilder();
			int c;
			while ((c = reader.read()) != -1) {
				builder.append((char) c);
			}
			int ret = process.waitFor();
			if (ret != 0) {
				throw new InterruptedException(Integer.toString(ret));
			}
			text = builder.toString();
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}
		return text;
	}
}
