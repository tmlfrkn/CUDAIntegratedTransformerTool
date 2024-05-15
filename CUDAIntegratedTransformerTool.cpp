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

cl::opt<std::string> // command line option to specify the number of threads for modification
    ThreadsValue("threads",
                 cl::desc("Specify the value to replace THREADS with"),
                 cl::value_desc("value"), cl::init("32"));

cl::opt<int>
    ThreadReductionRatio("reduction-ratio", // command line option to specify thread number reduction ratio0
                         cl::desc("Specify the value for reduction for number of threads"),
                         cl::init(0));
cl::opt<bool>
    ConvertDoubleToFloat("convert-double-to-float", // command line option to convert double types to floats
                         cl::desc("Convert double variables to float."),
                         cl::init(false));

cl::opt<bool>
    ChangeKernelCallParameter("change-Kernel", // command line option allowing change of Kernel lauch statement parameters
                              cl::desc("Change the number of threads inside the block"),
                              cl::init(false));

cl::opt<int> KernelParamNum("kernelParam-num",                                             // command line option to specify which kernel lauch parameter to modify 1 => change first parameter
                            cl::desc("Specify which kernel parameter to modify (1 or 2)"), // 2 => change second parameter (by default block)
                            cl::init(2));

cl::opt<bool>
    ChangeDim3("dim3", // command line option for allowing dim3 declaration parameters to be changed
               cl::desc("Change the parameters of dim3 declarations"),
               cl::init(false));

cl::opt<int> NumDim3Changes("num-dim3-changes", // command line option that defines how many dim3 declarations should be modified
                            cl::desc("Specify the number of dim3 declarations to change"),
                            cl::value_desc("number"), cl::init(-1));

cl::opt<std::string> Change_Variable_Name("change-var-name", // command line option taking the name of the variable to be changed
                                          cl::desc("Name of the variable to be changed"));

cl::opt<bool> Change_Specific_Variable("change-specific",
                                       cl::desc("Change value of specific variable"), // boolean command line option for allowing specific variable name to be modified
                                       cl::init(false));


cl::opt<bool>
    removeSynchThread("remove_synch_thread_to_null", // command line option to replace _synchtread() function to "NULL()":
                         cl::desc("Replace _synchthread() function with NULL()"),
                         cl::init(false));
cl::opt<bool>
    compremoveSynchThread("remove_synch_thread_to_empty", // command line option to Replace _synchthread() function with empty string":
                         cl::desc("Replace _synchthread() function with empty string"),
                         cl::init(false));
                         
cl::opt<bool> replaceWithSyncWarp("replace-with-syncwarp",
                                  cl::desc("Replace __syncthreads() function calls with __syncwarp()"),
                                  cl::init(false));
                                  
cl::opt<bool> replaceAtomicAddFunctiontoBlock("atomic-add-to-atomic-add-block",
                                  cl::desc("Replace atomicAdd() function with atomicAddBlock()"),
                                  cl::init(false));                    

cl::opt<bool> replaceAtomicAddFunctiontoDirect("atomic-to-direct",
                                  cl::desc("Replace atomicAdd() function with direct operation"),
                                  cl::init(false));
cl::opt<bool> ConvertIfElseToIfBody("convert-if-else-to-if-body",
                                    cl::desc("Convert if-else statements to only if-body."),
                                    cl::init(false));


class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor>
{
public:
    explicit MyASTVisitor(ASTContext *Context, Rewriter &R)
        : Context(Context), R(R) {}
        
        
    bool ModifyKernelElseBlock(CallExpr *CE)
{
    if(ConvertIfElseToIfBody){
    if (isCUDAKernelCall(CE))
    {
        SourceLocation StartLoc = CE->getCallee()->getBeginLoc();
        SourceLocation EndLoc = CE->getRParenLoc().getLocWithOffset(1);

        if (!StartLoc.isInvalid() && !EndLoc.isInvalid())
        {
            StringRef CallText = Lexer::getSourceText(CharSourceRange::getCharRange(StartLoc, EndLoc),
                                                      Context->getSourceManager(), Context->getLangOpts());

            size_t ifPos = CallText.find("if");
            size_t elsePos = CallText.find("else");

            if (ifPos != StringRef::npos && elsePos != StringRef::npos)
            {
           
                size_t lastClosingBracePos = CallText.rfind('}', elsePos);

                std::string modifiedCallText = CallText.substr(0, lastClosingBracePos).str() + CallText.substr(elsePos).str();


                SourceRange ReplacementRange(StartLoc, EndLoc);
                R.ReplaceText(ReplacementRange, modifiedCallText);

                return true;
            }
        }
        }
    }
    return false;
}
    

    // Visit type declarations and replace double with float.
    bool VisitTypeLoc(TypeLoc TL)
    {
        if (ConvertDoubleToFloat)
        {
            QualType QT = TL.getType();
            if (QT->isSpecificBuiltinType(BuiltinType::Double))
            {
                // Replace the type "double" with "float".
                SourceRange ReplacementRange = TL.getSourceRange();
                StringRef ReplacementText = "float";
                size_t OriginalLength = R.getRangeSize(ReplacementRange);
                size_t NewLength = ReplacementText.size();
                if (OriginalLength != NewLength)
                {
                    SourceLocation EndLoc = ReplacementRange.getBegin().getLocWithOffset(
                        OriginalLength - NewLength);
                    ReplacementRange.setEnd(EndLoc);
                }
                R.ReplaceText(ReplacementRange, ReplacementText);
            }
        }
        return true;
    }
    
     bool _synchThreadtoNull(CallExpr *E)//This function is find the _synchThread() function call and replace with NULL()
    {
    
    if (removeSynchThread){//For option section it initialized `false`, but command `--remove_synch_thread_to_null=true` makes the function going to executed and replace __synchThread() to "NULL()"
        if (FunctionDecl *FD = E->getDirectCallee())
        {
            if (FD->getNameAsString() == "__syncthreads")
            {
                // Replace the "__syncthreads()" call with "NULL"
                SourceRange ReplacementRange = E->getSourceRange();
                StringRef ReplacementText = "NULL";
                size_t OriginalLength = R.getRangeSize(ReplacementRange);
                size_t NewLength = ReplacementText.size();

                if (OriginalLength != NewLength)
                {
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

     bool _synchThreadto_syncwarp(CallExpr *E)
{
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
    bool replaceAtomicAddFunction(CallExpr *E) {
    	if(replaceAtomicAddFunctiontoBlock){
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
   	if(replaceAtomicAddFunctiontoDirect){
   
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

	






    bool VisitVarDecl(VarDecl *VD)
    {

        /* Give only one option as true at a time
           because Rewriter can not handle multiple source code changes in same place at once */
        if (Change_Specific_Variable && VD->hasInit())
        {
            Expr *Init = VD->getInit();
            std::string InitStr = Lexer::getSourceText(CharSourceRange::getTokenRange(Init->getSourceRange()),
                                                       Context->getSourceManager(), Context->getLangOpts())
                                      .str();

            // Check if the initialization string contains the macro names
            if (InitStr.find(Change_Variable_Name) != std::string::npos)
            {
                // Replace Change_Variable_Name with the value from the command line ThreadsValue
                ReplaceTextInSource(Init->getSourceRange(), std::regex(Change_Variable_Name), ThreadsValue);
            }
        }

        static int dim3ChangesMade = 0;

        ASTContext &Ctx = *Context;
        // Ensure the variable type is dim3
        if (VD->getType().getAsString().find("dim3") != std::string::npos && ChangeDim3)
        {

            if (NumDim3Changes >= 0 && dim3ChangesMade >= NumDim3Changes)
            {
                return true; // finalize the traversal if number of
                             // dim3 changes become greater than the specified command line option `NumDim3Changes`
            }
            // Directly working with source text to determine the number of parameters in the initializer
            std::string initText = Lexer::getSourceText(CharSourceRange::getTokenRange(VD->getSourceRange()),
                                                        Ctx.getSourceManager(), Ctx.getLangOpts())
                                       .str();

            // Simple heuristic to count commas in the initialization part to infer the number of parameters
            size_t commaCount = std::count(initText.begin(), initText.end(), ',');

            if (commaCount == 1)
            { // Likely two parameters
                llvm::errs() << "Transforming dim3 declaration with likely two parameters: " << initText << "\n";

                // Construct the replacement text
                int threadValueInt = stoi(ThreadsValue);
                int reducedThreadValue = threadValueInt - (threadValueInt * ThreadReductionRatio) / 100;
                std::string newThreadsValue = to_string(reducedThreadValue);
                std::string newInitCode = "dim3 " + VD->getNameAsString() + "(" + newThreadsValue + ", " + newThreadsValue + ");";

                // Replace the text in the source
                SourceLocation startLoc = VD->getBeginLoc();
                SourceLocation endLoc = Lexer::getLocForEndOfToken(VD->getEndLoc(), 0, Ctx.getSourceManager(), Ctx.getLangOpts());
                R.ReplaceText(SourceRange(startLoc, endLoc), newInitCode);

                dim3ChangesMade++; // incrementing the number of changes made for dim3
            }
        }
        return true;
    }

    // Visit call expressions and change the second parameter in CUDA kernel calls.
    bool VisitCallExpr(CallExpr *CE)
    {
    	_synchThreadtoNull(CE);
    	_synchThreadtoEmpty(CE);
    	_synchThreadto_syncwarp(CE);
    	replaceAtomicAddFunction(CE);
    	replaceAtomicAddWithDirectOperation(CE);
    	ModifyKernelElseBlock(CE);
    	
    	
        if (ChangeKernelCallParameter && isCUDAKernelCall(CE))
        {
            // Locate the entire call expression.
            SourceLocation StartLoc = CE->getCallee()->getBeginLoc();       // Start of the kernel function name.
            SourceLocation EndLoc = CE->getRParenLoc().getLocWithOffset(1); // End of the kernel call

            if (!StartLoc.isInvalid() && !EndLoc.isInvalid())
            {
                StringRef CallText = Lexer::getSourceText(CharSourceRange::getTokenRange(StartLoc, EndLoc), Context->getSourceManager(), Context->getLangOpts());

                llvm::errs() << "Call text is: " << CallText << "\n";
                llvm::errs() << "Call text contains (<<<): " << CallText.contains("<<<") << "\n";
                if (CallText.contains("<<<") && CallText.contains(">>>"))
                {
                    llvm::errs() << "Inside the CUDA kernel call: "
                                 << "\n";
                    size_t KernelConfigStart = CallText.find("<<<") + 3;
                    size_t KernelConfigEnd = CallText.find(">>>", KernelConfigStart);

                    // Extract the kernel configuration parameters.
                    std::string KernelConfig = CallText.slice(KernelConfigStart, KernelConfigEnd).str();
                    size_t CommaPos = KernelConfig.find(',');

                    // Parse the grid and block parameters.
                    std::string gridParam = KernelConfig.substr(0, CommaPos);
                    std::string blockParam = KernelConfig.substr(CommaPos + 1);

                    // Prepare the new parameter value based on threads and reduction ratio.
                    int ThreadValueInt = stoi(ThreadsValue);
                    std::string NewThreadsValue = std::to_string(ThreadValueInt - (ThreadValueInt * ThreadReductionRatio) / 100);

                    // Decide which parameter to replace based on KernelParamNum.
                    std::string ReplacementConfig;
                    if (KernelParamNum == 1)
                    {
                        llvm::errs() << "Kernel call changed First parameter" << CallText << "\n";
                        ReplacementConfig = NewThreadsValue + "," + blockParam;
                    }
                    else if (KernelParamNum == 2)
                    {
                        llvm::errs() << "Kernel call changed Second parameter" << CallText << "\n";
                        ReplacementConfig = gridParam + "," + NewThreadsValue;
                    }
                    else
                    {
                        llvm::errs() << "Invalid KernelParamNum value: " << KernelParamNum << ". Expected 1 or 2.\n";
                        return false;
                    }

                    // Construct the replacement text.
                    std::string ReplacementText = "<<<" + ReplacementConfig + ">>>";

                    // Calculate the source range for the replacement.
                    SourceLocation ReplacementBegin = StartLoc.getLocWithOffset(KernelConfigStart - 3); // Adjust for the offset of "<<<".
                    SourceLocation ReplacementEnd = StartLoc.getLocWithOffset(KernelConfigEnd + 2);     // Adjust for the offset of ">>>".
                    SourceRange ReplacementRange(ReplacementBegin, ReplacementEnd);

                    // Perform the replacement.
                    R.ReplaceText(ReplacementRange, ReplacementText);
                }
            }
            return true;
        }
        return true;
    }

private:
    ASTContext *Context;
    Rewriter &R;

    bool isCUDAKernelCall(CallExpr *CE)
    {
        if (CE->getDirectCallee())
        {
            // Check if the function is called with "<<<" and ">>>".
            SourceLocation BeginLoc = CE->getExprLoc();
            SourceLocation EndLoc = CE->getRParenLoc();

            if (!BeginLoc.isInvalid() && !EndLoc.isInvalid())
            {
                StringRef CallText = Lexer::getSourceText(CharSourceRange::getCharRange(BeginLoc, EndLoc),
                                                          Context->getSourceManager(), Context->getLangOpts());

                // llvm::errs() << "Call text is: " << CallText << "\n";
                // llvm::outs() << "Any Call Expression: " << CallText << "\n";
                if (CallText.contains("<<<") && CallText.contains(">>>"))
                {
                    llvm::errs() << "Found CUDA kernel call: " << CallText << "\n";
                    // llvm::outs() << "Found CUDA Kernel Call: " << CallText << "\n";
                    return true;
                }
            }
        }
        return false;
    }

    void ReplaceTextInSource(SourceRange range, std::regex pattern, std::string newValue)
    {
        std::string SourceText = Lexer::getSourceText(CharSourceRange::getTokenRange(range),
                                                      Context->getSourceManager(), Context->getLangOpts())
                                     .str();

        std::string NewText = std::regex_replace(SourceText, pattern, newValue);

        if (SourceText != NewText)
        {
            R.ReplaceText(range, NewText);
        }
    }
};








class MyASTConsumer : public ASTConsumer
{
public:
    explicit MyASTConsumer(ASTContext *Context, Rewriter &R)
        : Visitor(Context, R), TheRewriter(R) {}

    void HandleTranslationUnit(ASTContext &Context) override
    {
        // dim3ChangesMade = 0; // Reset the dim3 change counter for each file
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        TheRewriter.getEditBuffer(Context.getSourceManager().getMainFileID())
            .write(llvm::outs());
    }

private:
    MyASTVisitor Visitor;
    Rewriter &TheRewriter;
};

class MyFrontendAction : public ASTFrontendAction
{
public:
    MyFrontendAction() {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &Compiler,
                                                   StringRef InFile) override
    {
        TheRewriter.setSourceMgr(Compiler.getSourceManager(),
                                 Compiler.getLangOpts());
        return std::make_unique<MyASTConsumer>(&Compiler.getASTContext(),
                                               TheRewriter);
    }

private:
    Rewriter TheRewriter;
};

int main(int argc, const char **argv)
{
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser)
    {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }

    CommonOptionsParser &op = ExpectedParser.get();
    ClangTool Tool(op.getCompilations(), op.getSourcePathList());

    auto Factory = newFrontendActionFactory<MyFrontendAction>();
    int Result = Tool.run(Factory.get());
    
    
    
    if (Result != 0)
    {
        llvm::errs() << "Error occurred while running the tool.\n";
        return Result;
    }

    return 0;
}
