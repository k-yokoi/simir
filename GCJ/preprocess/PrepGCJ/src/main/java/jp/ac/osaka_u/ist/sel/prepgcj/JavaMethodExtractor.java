package jp.ac.osaka_u.ist.sel.prepgcj;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import javax.swing.tree.ExpandVetoException;

import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.RuleContext;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeListener;
import org.antlr.v4.runtime.tree.ParseTreeWalker;
import org.antlr.v4.runtime.tree.RuleNode;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.antlr.v4.runtime.tree.TerminalNodeImpl;
import org.antlr.v4.runtime.tree.Tree;
import org.antlr.v4.runtime.tree.xpath.XPath;

public class JavaMethodExtractor {
	private Path path;
	private Map<String, Method> methodMap = new HashMap<>();

	public JavaMethodExtractor(Path path) {
		this.path = path;
	}

	public String extract() throws Exception {

		CharStream input = CharStreams.fromPath(path);
		JavaLexer lexer = new JavaLexer(input);
		CommonTokenStream tokens = new CommonTokenStream(lexer);
		JavaParser parser = new JavaParser(tokens);
		ParseTree tree = parser.compilationUnit();
		for (ParseTree t : XPath.findAll(tree, "//methodDeclaration", parser)) {
			Method method = getMethod(t);
			if (methodMap.containsKey(method.name)) {
				System.err.println("Same name method is defined");
				if (method.text.size() > methodMap.get(method.name).text.size()) {
					methodMap.remove(method.name);
					methodMap.put(method.name, method);
				}
			} else {
				methodMap.put(method.name, method);
			}
		}
		
		List<String> calls = callMethods("main", 5);
		StringJoiner joiner = new StringJoiner(" ");
		for (String call : calls) {
			methodMap.get(call).text.forEach(s -> joiner.add(s));
		}
		return joiner.toString();
	}

	private Method getMethod(ParseTree tree) {
		String methodName = null;
		List<String> methodText = null;
		List<Token> methodIdentifiers = null;
		final int childCount = tree.getChildCount();
		for (int i = 0; i < childCount; i++) {
			ParseTree t = tree.getChild(i);
			if (t instanceof RuleNode) {
				RuleContext context = (RuleContext) t;
				if (context.getRuleIndex() == JavaParser.RULE_methodBody) {
					List<Token> methodTokens = getTokens(t);
					methodText = getText(methodTokens);
					methodIdentifiers = methodTokens.stream().
							filter(token -> token.getType() == JavaLexer.IDENTIFIER)
							.collect(Collectors.toList());
				}
			} else if (t instanceof TerminalNode) {
				TerminalNode node = (TerminalNode) t;
				if (node.getSymbol().getType() == JavaLexer.IDENTIFIER)
					methodName = t.getText();
			}
		}
		return new Method(methodName, methodIdentifiers, methodText);

	}
	
	private List<Token> getTokens(ParseTree tree) {
		List<Token> list = new ArrayList<Token>();
		new ParseTreeWalker().walk(new TerminalNodeListener(list), tree);
		return list;
	}
	
	private List<String> getText(List<Token> tokens) {
		List<String> list = new ArrayList<String>();
		for (Token token : tokens) {
			if (token.getType() == JavaLexer.IDENTIFIER) {
				list.addAll(separateIdentifier(token.getText()));
			} else if (token.getText().matches("[a-zA-Z]+")) {
				list.add(token.getText().toLowerCase());
			}
		}
		return list;
	}

	private List<String> callMethods(String methodName, int depth) {
		// System.out.println("call : " + functionName);
		List<String> calls = new ArrayList<>();
		calls.add(methodName);
		for (Token token : methodMap.get(methodName).identifiers) {
			if (methodMap.containsKey(token.getText()) && depth > 0) {
				calls.addAll(callMethods(token.getText(), --depth));
			}
		}
		return calls;
	}

	private static final Pattern pattern = Pattern.compile("(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])");

	private static final Pattern snakePatern = Pattern.compile("(_|-|\\d)+");
	private static final Pattern camelPatern = Pattern.compile("(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])|\\d+");

	private static List<String> separateIdentifier(String identifier) {
		String[] strings;
		if (identifier.contains("_") || identifier.contains("-"))
			strings = snakePatern.split(identifier);
		else
			strings = camelPatern.split(identifier);

		List<String> list = new ArrayList<String>();
		for (String string : strings) {
			if (string.length() > 0)
				list.add(string.toLowerCase());
		}

		return list;
	}
}
