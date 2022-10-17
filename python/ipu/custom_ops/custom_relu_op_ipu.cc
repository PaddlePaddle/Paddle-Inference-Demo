// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/shapeinference.hpp>

#include <popops/ElementWise.hpp>

namespace CustomOperators {
const popart::OperatorIdentifier ReluId = {"custom.ops", "Relu", 1};
}  // namespace CustomOperators
namespace CustomGradOperators {
const popart::OperatorIdentifier ReluGradId = {
    "custom.ops", "ReluGrad", 1};
}  // namespace CustomGradOperators

class ReluOp;
class ReluOpx;
class ReluGradOpx;

class ReluGradOp : public popart::Op {
 public:
  explicit ReluGradOp(const ReluOp &fwdOp);

  std::unique_ptr<popart::Op> clone() const final {
    return std::make_unique<ReluGradOp>(*this);
  }
  void setup() final { outInfo(0) = inInfo(0); };

  const std::vector<popart::GradInOutMapper> &gradInputInfo() const;

  // The Grad Op has 1 output, which is the gradient of the only input
  const std::map<int, int> &gradOutToNonGradIn() const;

  bool requiresRandomSeed() const override { return false; }

  // an estimate of how valuable sub-graph matching will be
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

class ReluOp : public popart::Op {
 public:
  ReluOp(const popart::OperatorIdentifier &_opid,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<ReluOp>(*this);
  }

  void setup() final { outInfo(0) = inInfo(0); }

  std::vector<std::unique_ptr<popart::Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> upops;
    upops.emplace_back(new ReluGradOp(*this));
    return upops;
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }
};

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition ReluOpDef({OpDefinition::Inputs({{"input", T}}),
                                    OpDefinition::Outputs({{"output", T}}),
                                    OpDefinition::Attributes({})});

static popart::OpCreator<ReluOp> ReluOpCreator(
    popart::OpDefinitions({{CustomOperators::ReluId, ReluOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      return std::make_unique<ReluOp>(info.opid, info.settings);
    },
    true);
}  // namespace

static popart::RegisterShapeInferenceFunction ReluShapeInfer(
    CustomOperators::ReluId,
    [](popart::ShapeInferenceContext &ctx  // NO_LINT
    ) { ctx.outInfo(0) = ctx.inInfo(0); });

namespace pe = popops::expr;

class ReluOpx : public popart::popx::Opx {
 public:
  ReluOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<ReluOp>(op, {CustomOperators::ReluId});
  }

  void grow(poplar::program::Sequence &prog) const final {  // NOLINT
    popart::logging::ir::trace("start Growing ReluOpx");

    auto op = getOp<ReluOp>();

    poplar::Tensor input = getInTensor(0);

    auto expression = pe::Select(pe::Const(0.0f),
                                 pe::_1,
                                 pe::Lt(pe::_1, pe::Const(0.0f)));

    popops::mapInPlace(graph(),
                       expression,
                       {input},
                       prog,
                       debugContext("Relu"),
                       poplar::OptionFlags());

    setOutTensor(0, input);
  }
};

class ReluGradOpx : public popart::popx::Opx {
 public:
  ReluGradOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<ReluGradOp>(op, {CustomGradOperators::ReluGradId});
  }

  void grow(poplar::program::Sequence &prog) const final {  // NOLINT
    auto op = getOp<ReluGradOp>();

    poplar::Tensor grad = getInTensor(0);
    poplar::Tensor input = getInTensor(1);

    pe::Mul expression = pe::Mul(
        pe::Select(
            pe::Const(1.0f), pe::Const(0.0f), pe::Gt(pe::_2, pe::Const(0.0f))),
        pe::_1);

    auto output = popops::map(graph(),
                              expression,
                              {grad, input},
                              prog,
                              debugContext("ReluGrad"),
                              poplar::OptionFlags());

    setOutTensor(0, output);
  }
};

ReluGradOp::ReluGradOp(const ReluOp &fwdOp)
    : popart::Op(CustomGradOperators::ReluGradId, fwdOp.settings) {}

const std::vector<popart::GradInOutMapper> &ReluGradOp::gradInputInfo()
    const {
  static const std::vector<popart::GradInOutMapper> inInfo = {
      {0, 0, popart::GradOpInType::GradOut}, {1, 0, popart::GradOpInType::In}};
  return inInfo;
}

// The Grad Op has 1 output, which is the gradient of the only input
const std::map<int, int> &ReluGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {{0, 0}};
  return outInfo;
}

static popart::popx::OpxCreator<ReluOpx> ReluOpxCreator(
    {CustomOperators::ReluId});
static popart::popx::OpxCreator<ReluGradOpx> ReluGradOpxCreator(
    {CustomGradOperators::ReluGradId});
