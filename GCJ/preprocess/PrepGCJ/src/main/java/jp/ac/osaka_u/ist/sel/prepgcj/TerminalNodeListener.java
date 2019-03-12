package jp.ac.osaka_u.ist.sel.prepgcj;

import java.util.List;

import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ErrorNode;
import org.antlr.v4.runtime.tree.ParseTreeListener;
import org.antlr.v4.runtime.tree.TerminalNode;

public class TerminalNodeListener implements ParseTreeListener {
	List<Token> list;
	
	public TerminalNodeListener(List<Token> list) {
		this.list = list;
	}

	public void enterEveryRule(ParserRuleContext arg0) {
		// TODO Auto-generated method stub
		
	}

	public void exitEveryRule(ParserRuleContext arg0) {
		// TODO Auto-generated method stub
		
	}

	public void visitErrorNode(ErrorNode arg0) {
		// TODO Auto-generated method stub
		
	}

	public void visitTerminal(TerminalNode arg0) {
		list.add(arg0.getSymbol());
	}

}
