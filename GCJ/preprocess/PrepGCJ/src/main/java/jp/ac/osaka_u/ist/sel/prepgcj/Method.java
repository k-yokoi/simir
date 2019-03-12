package jp.ac.osaka_u.ist.sel.prepgcj;

import java.util.List;

import org.antlr.v4.runtime.Token;

public class Method {
	public final String name;
	public final List<Token> identifiers;
	public final List<String> text;

	public Method(String name, List<Token> identifiers, List<String> text) {
		this.name = name;
		this.identifiers = identifiers;
		this.text = text;
	}

}
