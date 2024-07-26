#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Basic/SourceLocation.h"
#include "string.h"
#include <string>
#include <regex>
#include <iostream>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;
using namespace std;

static llvm::cl::OptionCategory MyToolCategory("my-tool options");

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

cl::opt<std::string> ThreadsValue("threads", cl::desc("Specify the value to replace THREADS with"), cl::value_desc("value"), cl::init("32"));

cl::opt<int> ThreadReductionRatio("reduction-ratio", cl::desc("Specify the value for reduction for number of threads"), cl::init(0));
cl::opt<bool> ConvertDoubleToFloat("convert-double-to-float", cl::desc("Convert double variables to float."), cl::init(false));

cl::opt<bool> ChangeKernelCallParameter("change-Kernel", cl::desc("Change the number of threads inside the block"), cl::init(false));
cl::opt<int> KernelParamNum("kernelParam-num", cl::desc("Specify which kernel parameter to modify (1 or 2)"), cl::init(2));

cl::opt<bool> ChangeDim3("dim3", cl::desc("Change the parameters of dim3 declarations"), cl::init(false));
cl::opt<int> NumDim3Changes("num-dim3-changes", cl::desc("Specify the number of dim3 declarations to change"), cl::value_desc("number"), cl::init(-1));

cl::opt<std::string> Change_Variable_Name("change-var-name", cl::desc("Name of the variable to be changed"));
cl::opt<bool> Change_Specific_Variable("change-specific", cl::desc("Change value of specific variable"), cl::init(false));

cl::opt<bool> removeSynchThread("remove_synch_thread_to_null", cl::desc("Replace _synchthread() function with NULL()"), cl::init(false));
cl::opt<bool> compremoveSynchThread("remove_synch_thread_to_empty", cl::desc("Replace _synchthread() function with empty string"), cl::init(false));

cl::opt<bool> replaceWithSyncWarp("replace-with-syncwarp", cl::desc("Replace __syncthreads() function calls with __syncwarp()"), cl::init(false));
cl::opt<bool> replaceAtomicAddFunctiontoBlock("atomic-add-to-atomic-add-block", cl::desc("Replace atomicAdd() function with atomicAddBlock()"), cl::init(false));
cl::opt<bool> replaceAtomicAddFunctiontoDirect("atomic-to-direct", cl::desc("Replace atomicAdd() function with direct operation"), cl::init(false));

cl::opt<bool> ConvertIfElseToIfBody("convert-if-else-to-if-body", cl::desc("Convert if-else statements to only if-body."), cl::init(false));
cl::opt<bool> SimplifyIfStatements("simplify-if-statements", cl::desc("Simplify function bodies by keeping only the first if statement body."), cl::init(false));
cl::opt<bool> SimplifyElseStatements("simplify-else-statements", cl::desc("Simplify function bodies by keeping only the else statement body."), cl::init(false));
cl::opt<bool> SimplifyElseIfStatements("simplify-else-if-statements", cl::desc("Simplify function bodies by keeping only the else if statement body."), cl::init(false));

cl::opt<int> IfIndex("if-index", cl::desc("Specify the index of the if statement occurrence to modify"), cl::value_desc("index"), cl::init(-1));
cl::opt<int> ElseIndex("else-index", cl::desc("Specify the index of the else statement occurrence to modify"), cl::value_desc("index"), cl::init(-1));
cl::opt<int> ElseIfIndex("elseif-index", cl::desc("Specify the index of the else if statement occurrence to modify"), cl::value_desc("index"), cl::init(-1));

cl::opt<int> SyncThreadsIndex("sync-index", cl::desc("Specify the index of the __syncthreads() occurrence to modify"), cl::value_desc("index"), cl::init(-1));
cl::opt<int> AtomicAddIndex("atomic-index", cl::desc("Specify the index of the atomicAdd() occurrence to modify"), cl::value_desc("index"), cl::init(-1));
cl::opt<int> DoubleIndex("double-index", cl::desc("Specify the index of the double variable occurrence to modify"), cl::value_desc("index"), cl::init(-1));

cl::opt<bool> synchcooperative("synchcooperative", cl::desc("Replace __syncthreads() with cooperative_groups::thread_group tile4_1 = cooperative_groups::tiled_partition(tile32_1, 4); tile4_1.sync();"), cl::init(false));
cl::opt<bool> synchactive("synchactive", cl::desc("Replace __syncthreads() with cooperative_groups::thread_group active1 = cooperative_groups::coalesced_threads(); active1.sync();"), cl::init(false));

class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor>
{
public:
    explicit MyASTVisitor(ASTContext *Context, Rewriter &R)
        : Context(Context), R(R), currentIfIndex(0), currentElseIndex(0), currentElseIfIndex(0),
          currentSyncIndex(0), currentAtomicIndex(0), currentDoubleIndex(0) {}

    bool VisitFunctionDecl(FunctionDecl *FD) {
        if (SimplifyIfStatements || SimplifyElseStatements || SimplifyElseIfStatements) {
            if (FD->hasAttr<CUDAGlobalAttr>()) {  // Check if it's a CUDA kernel
                CompoundStmt *Body = dyn_cast<CompoundStmt>(FD->getBody());
                if (Body && !Body->body_empty()) {
                    for (auto *item : Body->body()) {
                        if (isa<IfStmt>(item)) {  // Find the 'if' statement
                            currentIfIndex++;
                            if (currentIfIndex == IfIndex) {
                                simplifyIfStatement(FD, item);
                            }
                            // Check for else and else-if statements
                            if (SimplifyElseStatements) {
                                currentElseIndex++;
                                if (currentElseIndex == ElseIndex) {
                                    simplifyElseStatement(FD, item);
                                }
                            }
                            if (SimplifyElseIfStatements) {
                                currentElseIfIndex++;
                                if (currentElseIfIndex == ElseIfIndex) {
                                    simplifyElseIfStatement(FD, item);
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

    bool VisitCallExpr(CallExpr *E) {
        if (removeSynchThread || compremoveSynchThread || replaceWithSyncWarp || replaceAtomicAddFunctiontoBlock || replaceAtomicAddFunctiontoDirect || synchcooperative || synchactive) {
            if (FunctionDecl *FD = E->getDirectCallee()) {
                if (FD->getNameAsString() == "__syncthreads") {
                    currentSyncIndex++;
                    if (currentSyncIndex == SyncThreadsIndex) {
                        _synchThreadtoNull(E);
                        _synchThreadtoEmpty(E);
                        _synchThreadto_syncwarp(E);
                        _synchThreadtoCooperative(E);
                        _synchThreadtoActive(E);
                    }
                } else if (FD->getNameAsString() == "atomicAdd") {
                    currentAtomicIndex++;
                    if (currentAtomicIndex == AtomicAddIndex) {
                        replaceAtomicAddFunction(E);
                        replaceAtomicAddWithDirectOperation(E);
                    }
                }
            }
        }
        return true;
    }

    bool VisitTypeLoc(TypeLoc TL) {
        if (ConvertDoubleToFloat) {
            currentDoubleIndex++;
            if (currentDoubleIndex == DoubleIndex) {
                QualType QT = TL.getType();
                if (QT->isSpecificBuiltinType(BuiltinType::Double)) {
                    // Replace the type "double" with "float".
                    SourceRange ReplacementRange = TL.getSourceRange();
                    StringRef ReplacementText = "float";
                    size_t OriginalLength = R.getRangeSize(ReplacementRange);
                    size_t NewLength = ReplacementText.size();
                    if (OriginalLength != NewLength) {
                        SourceLocation EndLoc = ReplacementRange.getBegin().getLocWithOffset(
                            OriginalLength - NewLength);
                        ReplacementRange.setEnd(EndLoc);
                    }
                    R.ReplaceText(ReplacementRange, ReplacementText);
                }
            }
        }
        return true;
    }

private:
    ASTContext *Context;
    Rewriter &R;
    int currentIfIndex, currentElseIndex, currentElseIfIndex, currentSyncIndex, currentAtomicIndex, currentDoubleIndex;

    void simplifyIfStatement(FunctionDecl *FD, Stmt *item) {
        auto *IfStatement = cast<IfStmt>(item);
        SourceLocation IfStart = IfStatement->getIfLoc();

        // Copy the body of the 'if' excluding the braces
        StringRef IfBodyText = Lexer::getSourceText(
            CharSourceRange::getTokenRange(IfStatement->getThen()->getSourceRange()),
            Context->getSourceManager(), Context->getLangOpts(), 0);

        // Clear the entire function body and insert only the 'if' body
        SourceLocation EndLoc = Lexer::getLocForEndOfToken(FD->getBodyRBrace(), 0, Context->getSourceManager(), Context->getLangOpts());
        SourceRange FullFuncRange(IfStart, EndLoc);
        R.ReplaceText(FullFuncRange, IfBodyText);
    }

    void simplifyElseStatement(FunctionDecl *FD, Stmt *item) {
        auto *IfStatement = cast<IfStmt>(item);
        SourceLocation IfStart = IfStatement->getIfLoc();

        if (IfStatement->getElse()) { // Check if there's an 'else' statement
            Stmt *ElseStatement = IfStatement->getElse();

            // Copy the body of the 'else' excluding the braces
            StringRef ElseBodyText = Lexer::getSourceText(
                CharSourceRange::getTokenRange(ElseStatement->getSourceRange()),
                Context->getSourceManager(), Context->getLangOpts(), 0);

            // Clear the entire function body and insert only the 'else' body
            SourceLocation EndLoc = Lexer::getLocForEndOfToken(FD->getBodyRBrace(), 0, Context->getSourceManager(), Context->getLangOpts());
            SourceRange FullFuncRange(IfStart, EndLoc);
            R.ReplaceText(FullFuncRange, ElseBodyText);
        }
    }

    void simplifyElseIfStatement(FunctionDecl *FD, Stmt *item) {
        auto *IfStatement = cast<IfStmt>(item);

        // Traverse the 'if' chain to find 'else if'
        while (IfStatement) {
            if (auto *ElseIfStatement = dyn_cast<IfStmt>(IfStatement->getElse())) {
                IfStatement = ElseIfStatement;
                SourceLocation ElseIfStart = ElseIfStatement->getIfLoc();

                // Copy the body of the 'else if' excluding the braces
                StringRef ElseIfBodyText = Lexer::getSourceText(
                    CharSourceRange::getTokenRange(ElseIfStatement->getThen()->getSourceRange()),
                    Context->getSourceManager(), Context->getLangOpts(), 0);

                // Clear the entire function body and insert only the 'else if' body
                SourceLocation EndLoc = Lexer::getLocForEndOfToken(FD->getBodyRBrace(), 0, Context->getSourceManager(), Context->getLangOpts());
                SourceRange FullFuncRange(ElseIfStart, EndLoc);
                R.ReplaceText(FullFuncRange, ElseIfBodyText);

                break; // Only handle the first 'else if'
            } else {
                break;
            }
        }
    }

    bool _synchThreadtoNull(CallExpr *E) {
        if (removeSynchThread) {
            if (FunctionDecl *FD = E->getDirectCallee()) {
                if (FD->getNameAsString() == "__syncthreads") {
                    // Replace the "__syncthreads()" call with "NULL"
                    SourceRange ReplacementRange = E->getSourceRange();
                    StringRef ReplacementText = "NULL";
                    size_t OriginalLength = R.getRangeSize(ReplacementRange);
                    size_t NewLength = ReplacementText.size();

                    if (OriginalLength != NewLength) {
                        SourceLocation EndLoc = ReplacementRange.getBegin().getLocWithOffset(
                            OriginalLength - NewLength);
                        ReplacementRange.setEnd(EndLoc);
                    }
                    
                    R.ReplaceText(ReplacementRange, ReplacementText);
                }
            }
        }
        return true;
    }

    bool _synchThreadtoEmpty(CallExpr *E) {
        if (compremoveSynchThread) {
            if (FunctionDecl *FD = E->getDirectCallee()) {
                if (FD->getNameAsString() == "__syncthreads") {
                    SourceRange ReplacementRange = E->getSourceRange();

                    // Extend the range to include any subsequent semicolon
                    SourceLocation semicolonLoc = Lexer::findLocationAfterToken(ReplacementRange.getEnd(), tok::semi, Context->getSourceManager(), Context->getLangOpts(), false);
                    if (semicolonLoc.isValid()) {
                        ReplacementRange.setEnd(semicolonLoc);
                    }

                    // Remove the text by replacing it with an empty string
                    R.ReplaceText(ReplacementRange, "");
                }
            }
        }
        return true;
    }

    bool _synchThreadto_syncwarp(CallExpr *E) {
        if (replaceWithSyncWarp) {
            if (FunctionDecl *FD = E->getDirectCallee()) {
                if (FD->getNameAsString() == "__syncthreads") {
                    // Replace the "__syncthreads()" call with "_syncwarp()"
                    SourceRange ReplacementRange = E->getSourceRange();
                    StringRef ReplacementText = "__syncwarp";
                    size_t OriginalLength = R.getRangeSize(ReplacementRange);
                    size_t NewLength = ReplacementText.size();

                    if (OriginalLength != NewLength) {
                        SourceLocation EndLoc = ReplacementRange.getBegin().getLocWithOffset(
                            OriginalLength - NewLength);
                        ReplacementRange.setEnd(EndLoc);
                    }
                    
                    R.ReplaceText(ReplacementRange, ReplacementText);
                }
            }
        }
        return true;
    }

    bool _synchThreadtoCooperative(CallExpr *E) {
        if (synchcooperative) {
            if (FunctionDecl *FD = E->getDirectCallee()) {
                if (FD->getNameAsString() == "__syncthreads") {
                    // Replace the "__syncthreads()" call with the cooperative groups synchronization
                    SourceRange ReplacementRange = E->getSourceRange();
                    std::string ReplacementText = "cooperative_groups::thread_group tile32_1 = cooperative_groups::tiled_partition(cooperative_groups::this_thread_block(), 32);\n";
                    ReplacementText += "cooperative_groups::thread_group tile4_1 = cooperative_groups::tiled_partition(tile32_1, 4);\n";
                    ReplacementText += "tile4_1.sync()";

                    R.ReplaceText(ReplacementRange, ReplacementText);
                }
            }
        }
        return true;
    }

    bool _synchThreadtoActive(CallExpr *E) {
        if (synchactive) {
            if (FunctionDecl *FD = E->getDirectCallee()) {
                if (FD->getNameAsString() == "__syncthreads") {
                    // Replace the "__syncthreads()" call with the active threads synchronization
                    SourceRange ReplacementRange = E->getSourceRange();
                    std::string ReplacementText = "cooperative_groups::thread_group active1 = cooperative_groups::coalesced_threads();\n";
                    ReplacementText += "active1.sync()";

                    R.ReplaceText(ReplacementRange, ReplacementText);
                }
            }
        }
        return true;
    }

    bool replaceAtomicAddFunction(CallExpr *E) {
        if (replaceAtomicAddFunctiontoBlock) {
            if (FunctionDecl *FD = E->getDirectCallee()) {
                if (FD->getNameAsString() == "atomicAdd") {
                    std::string newText = "atomicAdd_block(";

                    // Iterate over the arguments and append them to the newText string
                    for (unsigned i = 0; i < E->getNumArgs(); ++i) {
                        std::string argText = Lexer::getSourceText(CharSourceRange::getTokenRange(E->getArg(i)->getSourceRange()),
                                                                  Context->getSourceManager(), Context->getLangOpts()).str();
                        if (i > 0) {
                            newText += ", ";
                        }
                        newText += argText;
                    }
                    newText += ")";

                    // Replace the entire expression with the new function call
                    SourceRange range = E->getSourceRange();
                    R.ReplaceText(range, newText);

                    return true; 
                }
            }
        }
        return false; // No replacement was made
    }

    bool replaceAtomicAddWithDirectOperation(CallExpr *CE) {
        if (replaceAtomicAddFunctiontoDirect) {
            if (FunctionDecl *FD = CE->getDirectCallee()) {
                if (FD->getNameAsString() == "atomicAdd") {
                    // Assuming atomicAdd has exactly two arguments
                    if (CE->getNumArgs() != 2) {
                        return false; // Safety check
                    }

                    Expr *ptrExpr = CE->getArg(0);
                    Expr *valueExpr = CE->getArg(1);

                    // Get the source text for the pointer expression and value expression
                    std::string ptrText = Lexer::getSourceText(CharSourceRange::getTokenRange(ptrExpr->getSourceRange()), 
                                                                Context->getSourceManager(), Context->getLangOpts()).str();
                    std::string valueText = Lexer::getSourceText(CharSourceRange::getTokenRange(valueExpr->getSourceRange()), 
                                                                 Context->getSourceManager(), Context->getLangOpts()).str();

                    // Generate the new text to replace the atomicAdd call
                    std::string newText;
                    newText += ptrText + ";\n";  // Assign previous value
                    newText += ptrText + " += " + valueText + ";\n"; // Update value

                    // Find where to insert the new text
                    SourceLocation startLoc = CE->getBeginLoc();
                    SourceLocation endLoc = Lexer::getLocForEndOfToken(CE->getEndLoc(), 0, Context->getSourceManager(), Context->getLangOpts());

                    // Replace the original atomicAdd call with the new lines of code
                    R.ReplaceText(SourceRange(startLoc, endLoc), newText);

                    return true;
                }
            }
        }
        return false;
    }
};

class MyASTConsumer : public ASTConsumer {
public:
    explicit MyASTConsumer(ASTContext *Context, Rewriter &R)
        : Visitor(Context, R), TheRewriter(R) {}

    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        TheRewriter.getEditBuffer(Context.getSourceManager().getMainFileID()).write(llvm::outs());
    }

private:
    MyASTVisitor Visitor;
    Rewriter &TheRewriter;
};

class MyFrontendAction : public ASTFrontendAction {
public:
    MyFrontendAction() {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler, StringRef InFile) override {
        TheRewriter.setSourceMgr(Compiler.getSourceManager(), Compiler.getLangOpts());
        return std::make_unique<MyASTConsumer>(&Compiler.getASTContext(), TheRewriter);
    }

private:
    Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }

    CommonOptionsParser &op = ExpectedParser.get();
    ClangTool Tool(op.getCompilations(), op.getSourcePathList());

    auto Factory = newFrontendActionFactory<MyFrontendAction>();
    int Result = Tool.run(Factory.get());
    
    if (Result != 0) {
        llvm::errs() << "Error occurred while running the tool.\n";
        return Result;
    }

    return 0;
}

