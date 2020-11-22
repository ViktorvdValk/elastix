/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// First include the header file to be tested:
#include "elxTransformIO.h"

#include "elxElastixTemplate.h"
#include "AdvancedAffineTransform/elxAdvancedAffineTransform.h"
#include "AdvancedBSplineTransform/elxAdvancedBSplineTransform.h"
#include "EulerTransform/elxEulerTransform.h"
#include "SimilarityTransform/elxSimilarityTransform.h"
#include "TranslationTransform/elxTranslationTransform.h"

#include <itkImage.h>
#include <itkAffineTransform.h>
#include <itkBSplineTransform.h>
#include <itkEuler2DTransform.h>
#include <itkEuler3DTransform.h>
#include <itkSimilarity2DTransform.h>
#include <itkSimilarity3DTransform.h>
#include <itkTranslationTransform.h>

#include <typeinfo>
#include <type_traits> // For is_same

#include <gtest/gtest.h>


namespace
{

template <unsigned NDimension>
using ElastixType = elx::ElastixTemplate<itk::Image<float, NDimension>, itk::Image<float, NDimension>>;

template <unsigned NDimension,
          template <typename>
          typename TElastixTransform,
          typename TExpectedCorrespondingItkTransform>
void
Expect_CorrespondingItkTransform()
{
  const auto elxTransform = TElastixTransform<ElastixType<NDimension>>::New();
  const auto itkTransform = elx::TransformIO::CreateCorrespondingItkTransform(*elxTransform);
  ASSERT_NE(itkTransform, nullptr);

  const auto & actualItkTransformTypeId = typeid(*itkTransform);
  const auto & expectedItkTransformTypeId = typeid(TExpectedCorrespondingItkTransform);
  ASSERT_EQ(std::string(actualItkTransformTypeId.name()), std::string(expectedItkTransformTypeId.name()));
  EXPECT_EQ(actualItkTransformTypeId, expectedItkTransformTypeId);
}


template <unsigned NDimension, template <typename> typename TElastixTransform>
void
Expect_default_elastix_FixedParameters_are_zero_for_specified_dimension()
{
  using TransformType = TElastixTransform<ElastixType<NDimension>>;
  const auto fixedParameters = TransformType::New()->GetFixedParameters();
  ASSERT_EQ(fixedParameters, vnl_vector<double>(fixedParameters.size(), 0.0));
}

template <unsigned NDimension>
void
Expect_default_AdvancedBSplineTransform_GetParameters_throws_ExceptionObject(const bool fixed)
{
  const auto transform = elx::AdvancedBSplineTransform<ElastixType<NDimension>>::New();
  EXPECT_THROW(elx::TransformIO::GetParameters(fixed, *transform), itk::ExceptionObject);
}


template <unsigned NDimension, template <typename> typename TElastixTransform>
void
Expect_default_elastix_Parameters_remain_the_same_when_set_for_specified_dimension(const bool fixed)
{
  using TransformType = TElastixTransform<ElastixType<NDimension>>;

  const auto transform = TransformType::New();
  const auto parameters = elx::TransformIO::GetParameters(fixed, *transform);
  elx::TransformIO::SetParameters(fixed, *transform, parameters);
  ASSERT_EQ(elx::TransformIO::GetParameters(fixed, *transform), parameters);
}


template <template <typename> typename TElastixTransform>
void
Expect_default_elastix_FixedParameters_are_zero()
{
  Expect_default_elastix_FixedParameters_are_zero_for_specified_dimension<2, TElastixTransform>();
  Expect_default_elastix_FixedParameters_are_zero_for_specified_dimension<3, TElastixTransform>();
  Expect_default_elastix_FixedParameters_are_zero_for_specified_dimension<4, TElastixTransform>();
}


template <template <typename> typename TElastixTransform>
void
Expect_default_elastix_Parameters_remain_the_same_when_set(const bool fixed)
{
  Expect_default_elastix_Parameters_remain_the_same_when_set_for_specified_dimension<2, TElastixTransform>(fixed);
  Expect_default_elastix_Parameters_remain_the_same_when_set_for_specified_dimension<3, TElastixTransform>(fixed);
  Expect_default_elastix_Parameters_remain_the_same_when_set_for_specified_dimension<4, TElastixTransform>(fixed);
}


template <unsigned NDimension,
          template <typename>
          typename TElastixTransform,
          typename TExpectedCorrespondingItkTransform>
void
Test_copying_default_parameters(const bool fixed)
{
  const auto elxTransform = TElastixTransform<ElastixType<NDimension>>::New();
  SCOPED_TRACE(elxTransform->elxGetClassName());

  const auto itkTransform = elx::TransformIO::CreateCorrespondingItkTransform(*elxTransform);
  ASSERT_NE(itkTransform, nullptr);

  const auto parameters = elx::TransformIO::GetParameters(fixed, *elxTransform);
  elx::TransformIO::SetParameters(fixed, *itkTransform, parameters);

  ASSERT_EQ(elx::TransformIO::GetParameters(fixed, *itkTransform), parameters);
}


template <unsigned NDimension,
          template <typename>
          typename TElastixTransform,
          typename TExpectedCorrespondingItkTransform>
void
Test_copying_parameters()
{
  const auto elxTransform = TElastixTransform<ElastixType<NDimension>>::New();

  SCOPED_TRACE(elxTransform->elxGetClassName());

  const auto itkTransform = elx::TransformIO::CreateCorrespondingItkTransform(*elxTransform);
  ASSERT_NE(itkTransform, nullptr);

  const auto & actualItkTransformTypeId = typeid(*itkTransform);
  const auto & expectedItkTransformTypeId = typeid(TExpectedCorrespondingItkTransform);
  ASSERT_EQ(std::string(actualItkTransformTypeId.name()), std::string(expectedItkTransformTypeId.name()));
  EXPECT_EQ(actualItkTransformTypeId, expectedItkTransformTypeId);

  auto parameters = elxTransform->GetParameters();
  std::iota(std::begin(parameters), std::end(parameters), 1.0);
  std::for_each(std::begin(parameters), std::end(parameters), [](double & x) { x /= 8; } );
  elxTransform->SetParameters(parameters);
  ASSERT_EQ(elxTransform->GetParameters(), parameters);

  auto fixedParameters = elxTransform->GetFixedParameters();
  std::iota(std::begin(fixedParameters), std::end(fixedParameters), 1.0);
  elxTransform->SetFixedParameters(fixedParameters);
  ASSERT_EQ(elxTransform->GetFixedParameters(), fixedParameters);

  itkTransform->SetParameters(parameters);
  itkTransform->SetFixedParameters(fixedParameters);

  ASSERT_EQ(itkTransform->GetParameters(), parameters);
  ASSERT_EQ(itkTransform->GetFixedParameters(), fixedParameters);
}

} // namespace


GTEST_TEST(TransformIO, CorrespondingItkTransform)
{
  Expect_CorrespondingItkTransform<2, elx::AdvancedAffineTransformElastix, itk::AffineTransform<double, 2>>();
  Expect_CorrespondingItkTransform<3, elx::AdvancedAffineTransformElastix, itk::AffineTransform<double, 3>>();
  Expect_CorrespondingItkTransform<4, elx::AdvancedAffineTransformElastix, itk::AffineTransform<double, 4>>();
  
  Expect_CorrespondingItkTransform<2, elx::AdvancedBSplineTransform, itk::BSplineTransform<double, 2>>();
  Expect_CorrespondingItkTransform<3, elx::AdvancedBSplineTransform, itk::BSplineTransform<double, 3>>();

  Expect_CorrespondingItkTransform<2, elx::TranslationTransformElastix, itk::TranslationTransform<double, 2>>();
  Expect_CorrespondingItkTransform<3, elx::TranslationTransformElastix, itk::TranslationTransform<double, 3>>();
  Expect_CorrespondingItkTransform<4, elx::TranslationTransformElastix, itk::TranslationTransform<double, 4>>();

  Expect_CorrespondingItkTransform<2, elx::SimilarityTransformElastix, itk::Similarity2DTransform<double>>();
  Expect_CorrespondingItkTransform<3, elx::SimilarityTransformElastix, itk::Similarity3DTransform<double>>();

  Expect_CorrespondingItkTransform<2, elx::EulerTransformElastix, itk::Euler2DTransform<double>>();
  Expect_CorrespondingItkTransform<3, elx::EulerTransformElastix, itk::Euler3DTransform<double>>();
}


GTEST_TEST(TransformIO, DefaultAdvancedBSplineTransformGetParametersThrowsExceptionObject)
{
  for (const bool fixed : { false, true })
  {
    Expect_default_AdvancedBSplineTransform_GetParameters_throws_ExceptionObject<2>(fixed);
    Expect_default_AdvancedBSplineTransform_GetParameters_throws_ExceptionObject<3>(fixed);
  }
}


GTEST_TEST(TransformIO, DefaultElastixFixedParametersAreZero)
{
  using namespace elastix;

  // Note: This test would fail for AdvancedBSplineTransform, which is related to the test
  // DefaultAdvancedBSplineTransformGetParametersThrowsExceptionObject.
  Expect_default_elastix_FixedParameters_are_zero<AdvancedAffineTransformElastix>();
  Expect_default_elastix_FixedParameters_are_zero<EulerTransformElastix>();
  Expect_default_elastix_FixedParameters_are_zero<SimilarityTransformElastix>();
  Expect_default_elastix_FixedParameters_are_zero<TranslationTransformElastix>();
}


GTEST_TEST(TransformIO, DefaultElastixParametersRemainTheSameWhenSet)
{
  for (const bool fixed : { false, true })
  {
    using namespace elastix;

    // Note: This test would fail for AdvancedBSplineTransform, which is related to the test
    // DefaultAdvancedBSplineTransformGetParametersThrowsExceptionObject.

    Expect_default_elastix_Parameters_remain_the_same_when_set<AdvancedAffineTransformElastix>(fixed);
    Expect_default_elastix_Parameters_remain_the_same_when_set<EulerTransformElastix>(fixed);
    Expect_default_elastix_Parameters_remain_the_same_when_set<SimilarityTransformElastix>(fixed);
    Expect_default_elastix_Parameters_remain_the_same_when_set<TranslationTransformElastix>(fixed);
  }
}


GTEST_TEST(TransformIO, CopyDefaultParametersToCorrespondingItkTransform)
{
  for (const bool fixed : { false, true })
  {
    // Note: This test would fail for elx::AdvancedBSplineTransform, which is related to the test
    // DefaultAdvancedBSplineTransformGetParametersThrowsExceptionObject.

    Test_copying_default_parameters<2, elx::AdvancedAffineTransformElastix, itk::AffineTransform<double, 2>>(fixed);
    Test_copying_default_parameters<3, elx::AdvancedAffineTransformElastix, itk::AffineTransform<double, 3>>(fixed);
    Test_copying_default_parameters<4, elx::AdvancedAffineTransformElastix, itk::AffineTransform<double, 4>>(fixed);

    Test_copying_default_parameters<2, elx::TranslationTransformElastix, itk::TranslationTransform<double, 2>>(fixed);
    Test_copying_default_parameters<3, elx::TranslationTransformElastix, itk::TranslationTransform<double, 3>>(fixed);
    Test_copying_default_parameters<4, elx::TranslationTransformElastix, itk::TranslationTransform<double, 4>>(fixed);

    Test_copying_default_parameters<2, elx::SimilarityTransformElastix, itk::Similarity2DTransform<double>>(fixed);
    Test_copying_default_parameters<3, elx::SimilarityTransformElastix, itk::Similarity3DTransform<double>>(fixed);
    Test_copying_default_parameters<2, elx::EulerTransformElastix, itk::Euler2DTransform<double>>(fixed);

    Test_copying_default_parameters<3, elx::EulerTransformElastix, itk::Euler2DTransform<double>>(fixed);
  }
}


GTEST_TEST(TransformIO, CopyParametersToCorrespondingItkTransform)
{
  Test_copying_parameters<2, elx::AdvancedAffineTransformElastix, itk::AffineTransform<double, 2>>();
  Test_copying_parameters<3, elx::AdvancedAffineTransformElastix, itk::AffineTransform<double, 3>>();
  Test_copying_parameters<2, elx::TranslationTransformElastix, itk::TranslationTransform<double, 2>>();
  Test_copying_parameters<3, elx::TranslationTransformElastix, itk::TranslationTransform<double, 3>>();

  Test_copying_parameters<2, elx::SimilarityTransformElastix, itk::Similarity2DTransform<double>>();
  Test_copying_parameters<3, elx::SimilarityTransformElastix, itk::Similarity3DTransform<double>>();
  Test_copying_parameters<2, elx::EulerTransformElastix, itk::Euler2DTransform<double>>();
}
