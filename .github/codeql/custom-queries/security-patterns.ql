/**
 * @name Custom security patterns for FinChat-SEC-QA
 * @description Detects application-specific security patterns and vulnerabilities
 * @kind problem
 * @problem.severity warning
 * @security-severity 7.5
 * @precision high
 * @id finchat/security-patterns
 * @tags security
 *       external/cwe/cwe-020
 *       external/cwe/cwe-077
 *       external/cwe/cwe-089
 */

import python
import semmle.python.security.dataflow.SqlInjectionQuery
import semmle.python.security.dataflow.CodeInjectionQuery

// Detect potential SQL injection in financial data queries
class FinancialSqlInjection extends SqlInjection {
  FinancialSqlInjection() {
    exists(Call call |
      call.getFunc().(Name).getId() in ["execute", "executemany", "query"] and
      call.getAnArg().toString().matches("%ticker%") and
      this.asExpr() = call.getAnArg()
    )
  }
}

// Detect hardcoded API keys or secrets
class HardcodedSecrets extends Expr {
  HardcodedSecrets() {
    exists(StringLiteral s |
      s = this and
      (
        s.getText().matches("%api_key%") or
        s.getText().matches("%secret%") or
        s.getText().matches("%token%") or
        s.getText().matches("%password%")
      ) and
      not s.getLocation().getFile().getAbsolutePath().matches("%test%")
    )
  }
}

// Detect unsafe file operations with user input
class UnsafeFileOperation extends Call {
  UnsafeFileOperation() {
    this.getFunc().(Name).getId() in ["open", "read", "write", "remove", "unlink"] and
    exists(DataFlow::PathNode source, DataFlow::PathNode sink |
      source.getNode().asExpr() instanceof Name and
      sink.getNode().asExpr() = this.getAnArg() and
      DataFlow::hasFlow(source.getNode(), sink.getNode())
    )
  }
}

// Detect potential command injection in EDGAR processing
class EdgarCommandInjection extends Call {
  EdgarCommandInjection() {
    this.getFunc().(Attribute).getName() in ["system", "popen", "subprocess"] and
    exists(StringLiteral s |
      s = this.getAnArg() and
      s.getText().matches("%edgar%")
    )
  }
}

from Expr vulnerability, string message
where
  (
    vulnerability instanceof HardcodedSecrets and
    message = "Potential hardcoded secret or API key detected"
  ) or
  (
    vulnerability instanceof UnsafeFileOperation and
    message = "Unsafe file operation with potential user input"
  ) or
  (
    vulnerability instanceof EdgarCommandInjection and
    message = "Potential command injection in EDGAR processing"
  )
select vulnerability, message