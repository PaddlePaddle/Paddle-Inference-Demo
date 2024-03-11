#include "paddle/extension.h"

namespace {

class ReluReplacePattern : public paddle::drr::DrrPatternBase {
public:
  std::string name() const override { return "ReluReplacePattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &relu = pat.Op("pd_op.relu");
    relu({&pat.Tensor("in")}, {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &custom_relu = res.Op("custom_op.custom_relu");
    custom_relu({&res.Tensor("in")}, {&res.Tensor("out")});
  }
};

class ReluReplacePass : public pir::PatternRewritePass {
public:
  ReluReplacePass() : pir::PatternRewritePass("relu_replace_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<ReluReplacePattern>(context));
    return ps;
  }
};

} // namespace

REGISTER_IR_PASS(relu_replace_pass, ReluReplacePass);
