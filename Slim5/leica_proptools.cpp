/**
** Copyright (c) 2012 Leica Microsystems - All Rights Reserved
**
** Tool functions for AHM properties
**
**/
#include "stdafx.h"
#if (HAS_LEICA ||(BUILD_ALL_DEVICES_TARGETS))

#include <reuse/proptools.h>
#include <reuse/ahmtools.h>


//#define __DO_TEST__
#ifdef __DO_TEST__
#include <iostream>
#endif

namespace ahm {

	namespace property_tools {

#define throwException(cls,code,sztext)

		static ahm::Property* findProperty(ahm::Properties* pProperties, iop::int32 id) {
			return pProperties != 0 ? pProperties->findProperty(id) : 0;
		}

		ahm::Properties* properties(ahm::Unit* pUnit) {
			return find_itf_version<ahm::Properties>(pUnit, ahm::IID_PROPERTIES, ahm::Properties::INTERFACE_VERSION);
		}

		ahm::Properties* properties(ahm::Property* pProperty) {
			return pProperty != 0 ? properties(pProperty->value()) : 0;
		}

		ahm::Properties* properties(ahm::PropertyValue* pPropertyValue) {
			return pPropertyValue != 0 && pPropertyValue->derivedType() == ahm::PropertyValue::TYPE_PROPERTIES ? ((ahm::PropertyValueProperties*) pPropertyValue)->properties() : 0;
		}

		bool getIntValue(ahm::PropertyValue* pValue, iop::int32& iresult, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_INT:
				{
					ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
					iresult = pIntValue->getValue();
					return true;
				} break;

				case ahm::PropertyValue::TYPE_STRING:
					if (pConversion) {
						ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*)pValue;
						ahm::StringResult* pStrResult = pStringValue->getValue();
						if (pStrResult) {
							std::string strvalue;
							strvalue = pStrResult->value();
							pStrResult->dispose();
							return pConversion->convertStringToInt(strvalue.c_str(), iresult);
						}

					} break;
				case ahm::PropertyValue::TYPE_FLOAT:
					if (pConversion) {
						ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;
						return pConversion->convertFloatToInt(pFloatValue->getValue(), iresult);
					} break;
				case ahm::PropertyValue::TYPE_BOOL:
					if (pConversion) {
						ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
						return pConversion->convertBoolToInt(pBoolValue->getValue(), iresult);
					} break;
				case ahm::PropertyValue::TYPE_UNICODE_STRING:
					if (pConversion) {
						ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*) pValue;
						ahm::UnicodeStringResult* pUnicodeStringResult = pUnicodeStringValue->getValue();

						if (pUnicodeStringResult) {
							bool flagResult = pConversion->convertUnicodeStringToInt(pUnicodeStringResult->value(), iresult);
							pUnicodeStringResult->dispose();
							return flagResult;
						}
					}

					break;				// PropertyValueUnicodeString
				case ahm::PropertyValue::TYPE_INDEX:
				{// no conversion
					ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;
					iresult = pIndexValue->getIndex();
					return true;
				} break;
				}

			}
			return false;
		}
		bool setIntValue(ahm::PropertyValue* pValue, iop::int32 ivalue, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_INT:
				{
					ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
					pIntValue->setValue(ivalue);
					return true;

				} break;
				case ahm::PropertyValue::TYPE_STRING:
					if (pConversion) {
						ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*)pValue;
						std::string strval;
						if (pConversion->convertIntToString(ivalue, strval)) {
							pStringValue->setValue(strval.c_str());
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_FLOAT:
					if (pConversion) {
						ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;
						iop::int32 ival = 0;
						if (pConversion->convertFloatToInt(pFloatValue->getValue(), ivalue)) {
							pFloatValue->setValue(ival);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_BOOL:
					if (pConversion) {
						ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
						bool flag = false;
						if (pConversion->convertIntToBool(ivalue, flag)) {
							pBoolValue->setValue(flag);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_UNICODE_STRING:
					if (pConversion) {
						ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*) pValue;
						std::wstring wstr;
						if (pConversion->convertIntToUnicodeString(ivalue, wstr)) {
							pUnicodeStringValue->setValue(wstr.c_str());
							return true;
						}
					}
					break;
				case ahm::PropertyValue::TYPE_INDEX:
				{
					ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;
					pIndexValue->setIndex(ivalue);
					return true;
				}
				break;
				}
			}
			return false;
		}
		bool getIntValue(ahm::Property* pProperty, iop::int32& iresult, IValueConversion* pConversion /* = 0 */) {
			return pProperty ? getIntValue(pProperty->value(), iresult, pConversion) : false;
		}
		bool setIntValue(ahm::Property* pProperty, iop::int32 ivalue, IValueConversion* pConversion /* = 0 */) {
			return pProperty ? setIntValue(pProperty->value(), ivalue, pConversion) : false;
		}
		bool getIntValue(ahm::Properties* pProperties, iop::int32 id, iop::int32& iresult, IValueConversion* pConversion /* = 0 */) {
			return getIntValue(findProperty(pProperties, id), iresult, pConversion);
		}
		bool setIntValue(ahm::Properties* pProperties, iop::int32 id, iop::int32 ivalue, IValueConversion* pConversion /* = 0 */) {
			return setIntValue(findProperty(pProperties, id), ivalue, pConversion);
		}

		bool getFloatValue(ahm::PropertyValue* pValue, iop::float64& f64result, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_FLOAT:
				{
					ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;
					f64result = pFloatValue->getValue();
					return true;
				} break;

				case ahm::PropertyValue::TYPE_STRING:
					if (pConversion) {
						ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*)pValue;
						ahm::StringResult* pStrResult = pStringValue->getValue();
						if (pStrResult) {
							std::string strvalue;
							strvalue = pStrResult->value();
							pStrResult->dispose();
							return pConversion->convertStringToFloat(strvalue.c_str(), f64result);
						}

					} break;
				case ahm::PropertyValue::TYPE_INT:
					if (pConversion) {
						ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
						return pConversion->convertIntToFloat(pIntValue->getValue(), f64result);
					} break;
				case ahm::PropertyValue::TYPE_BOOL:
					if (pConversion) {
						ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
						return pConversion->convertBoolToFloat(pBoolValue->getValue(), f64result);
					} break;
				case ahm::PropertyValue::TYPE_UNICODE_STRING:
					if (pConversion) {
						ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*) pValue;
						ahm::UnicodeStringResult* pUnicodeStringResult = pUnicodeStringValue->getValue();

						if (pUnicodeStringResult) {
							bool flagResult = pConversion->convertUnicodeStringToFloat(pUnicodeStringResult->value(), f64result);
							pUnicodeStringResult->dispose();
							return flagResult;
						}
					}

					break;				// PropertyValueUnicodeString
				case ahm::PropertyValue::TYPE_INDEX:
					if (pConversion) {
						ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;

						return  pConversion->convertIntToFloat(pIndexValue->getIndex(), f64result);
					} break;
				}

			}
			return false;
		}

		bool setFloatValue(ahm::PropertyValue* pValue, iop::float64 f64value, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_FLOAT:
				{
					ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;

					pFloatValue->setValue(f64value);
					return true;
				} break;
				case ahm::PropertyValue::TYPE_STRING:
					if (pConversion) {
						ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*)pValue;
						std::string strval;
						if (pConversion->convertFloatToString(f64value, strval)) {
							pStringValue->setValue(strval.c_str());
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_INT:
					if (pConversion) {
						ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
						iop::int32 ival = 0;
						if (pConversion->convertFloatToInt(f64value, ival)) {
							pIntValue->setValue(ival);
							return true;
						}
					} break;

				case ahm::PropertyValue::TYPE_BOOL:
					if (pConversion) {
						ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
						bool flag = false;
						if (pConversion->convertFloatToBool(f64value, flag)) {
							pBoolValue->setValue(flag);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_UNICODE_STRING:
					if (pConversion) {
						ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*) pValue;
						std::wstring wstr;
						if (pConversion->convertFloatToUnicodeString(f64value, wstr)) {
							pUnicodeStringValue->setValue(wstr.c_str());
							return true;
						}
					}
					break;
				case ahm::PropertyValue::TYPE_INDEX:
					if (pConversion) {
						ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;

						iop::int32 ival = 0;
						if (pConversion->convertFloatToInt(f64value, ival)) {
							pIndexValue->setIndex(ival);
							return true;
						}
					}
					break;
				}
			}
			return false;
		}
		bool getFloatValue(ahm::Property* pProperty, iop::float64& f64result, IValueConversion* pConversion /* = 0 */) {
			return pProperty != 0 ? getFloatValue(pProperty->value(), f64result, pConversion) : false;
		}

		bool setFloatValue(ahm::Property* pProperty, iop::float64 f64value, IValueConversion* pConversion /* = 0 */) {
			return pProperty != 0 ? setFloatValue(pProperty->value(), f64value, pConversion) : false;
		}

		bool getFloatValue(ahm::Properties* pProperties, iop::int32 id, iop::float64& f64result, IValueConversion* pConversion /* = 0 */) {
			return getFloatValue(findProperty(pProperties, id), f64result, pConversion);
		}

		bool setFloatValue(ahm::Properties* pProperties, iop::int32 id, iop::float64 f64value, IValueConversion* pConversion /* = 0 */) {
			return setFloatValue(findProperty(pProperties, id), f64value, pConversion);
		}

		bool getBoolValue(ahm::PropertyValue* pValue, bool& bresult, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_BOOL:
				{
					ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
					bresult = pBoolValue->getValue();
					return true;
				} break;

				case ahm::PropertyValue::TYPE_FLOAT:
					if (pConversion) {
						ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;
						return pConversion->convertFloatToBool(pFloatValue->getValue(), bresult);
					} break;

				case ahm::PropertyValue::TYPE_STRING:
					if (pConversion) {
						ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*)pValue;
						ahm::StringResult* pStrResult = pStringValue->getValue();
						if (pStrResult) {
							std::string strvalue;
							strvalue = pStrResult->value();
							pStrResult->dispose();
							return pConversion->convertStringToBool(strvalue.c_str(), bresult);
						}

					} break;
				case ahm::PropertyValue::TYPE_INT:
					if (pConversion) {
						ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
						return pConversion->convertIntToBool(pIntValue->getValue(), bresult);
					} break;

				case ahm::PropertyValue::TYPE_UNICODE_STRING:
					if (pConversion) {
						ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*) pValue;
						ahm::UnicodeStringResult* pUnicodeStringResult = pUnicodeStringValue->getValue();

						if (pUnicodeStringResult) {
							bool flagResult = pConversion->convertUnicodeStringToBool(pUnicodeStringResult->value(), bresult);
							pUnicodeStringResult->dispose();
							return flagResult;
						}
					}

					break;				// PropertyValueUnicodeString
				case ahm::PropertyValue::TYPE_INDEX:
					if (pConversion) {
						ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;
						return  pConversion->convertIntToBool(pIndexValue->getIndex(), bresult);
					} break;
				}
			}
			return false;
		}
		bool setBoolValue(ahm::PropertyValue* pValue, bool bvalue, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_BOOL:
				{
					ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
					pBoolValue->setValue(bvalue);
					return true;
				} break;
				case ahm::PropertyValue::TYPE_STRING:
					if (pConversion) {
						ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*)pValue;
						std::string strval;
						if (pConversion->convertBoolToString(bvalue, strval)) {
							pStringValue->setValue(strval.c_str());
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_INT:
					if (pConversion) {
						ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
						iop::int32 ival = 0;
						if (pConversion->convertBoolToInt(bvalue, ival)) {
							pIntValue->setValue(ival);
							return true;
						}
					} break;

				case ahm::PropertyValue::TYPE_FLOAT:
					if (pConversion) {
						ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;
						iop::float64 f64val = 0.0;
						if (pConversion->convertBoolToFloat(bvalue, f64val)) {
							pFloatValue->setValue(f64val);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_UNICODE_STRING:
					if (pConversion) {
						ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*) pValue;
						std::wstring wstr;
						if (pConversion->convertBoolToUnicodeString(bvalue, wstr)) {
							pUnicodeStringValue->setValue(wstr.c_str());
							return true;
						}
					}
					break;
				case ahm::PropertyValue::TYPE_INDEX:
					if (pConversion) {
						ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;

						iop::int32 ival = 0;
						if (pConversion->convertBoolToInt(bvalue, ival)) {
							pIndexValue->setIndex(ival);
							return true;
						}
					}
					break;
				}
			}
			return false;
		}
		bool getBoolValue(ahm::Property* pProperty, bool& bresult, IValueConversion* pConversion /* = 0 */) {
			return pProperty != 0 ? getBoolValue(pProperty->value(), bresult, pConversion) : false;
		}

		bool setBoolValue(ahm::Property* pProperty, bool bvalue, IValueConversion* pConversion /* = 0 */) {
			return pProperty != 0 ? setBoolValue(pProperty->value(), bvalue, pConversion) : false;
		}
		bool getBoolValue(ahm::Properties* pProperties, iop::int32 id, bool& bresult, IValueConversion* pConversion /* = 0 */) {
			return getBoolValue(findProperty(pProperties, id), bresult, pConversion);
		}
		bool setBoolValue(ahm::Properties* pProperties, iop::int32 id, bool bvalue, IValueConversion* pConversion /* = 0 */) {
			return setBoolValue(findProperty(pProperties, id), bvalue, pConversion);
		}

		bool getStringValue(ahm::PropertyValue* pValue, std::string& strresult, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_STRING:
				{
					ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*)pValue;
					ahm::StringResult* pStrResult = pStringValue->getValue();
					if (pStrResult) {
						strresult = pStrResult->value();
						pStrResult->dispose();
						return true;
					}
				} break;
				case ahm::PropertyValue::TYPE_INT:
					if (pConversion) {
						ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
						return pConversion->convertIntToString(pIntValue->getValue(), strresult);
					} break;
				case ahm::PropertyValue::TYPE_FLOAT:
					if (pConversion) {
						ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;
						return pConversion->convertFloatToString(pFloatValue->getValue(), strresult);
					} break;
				case ahm::PropertyValue::TYPE_BOOL:
					if (pConversion) {
						ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
						return pConversion->convertBoolToString(pBoolValue->getValue(), strresult);

					} break;
				case ahm::PropertyValue::TYPE_UNICODE_STRING:
					if (pConversion) {
						ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*) pValue;
						ahm::UnicodeStringResult* pUnicodeStringResult = pUnicodeStringValue->getValue();

						if (pUnicodeStringResult) {
							bool flagResult = pConversion->convertUnicodeStringToString(pUnicodeStringResult->value(), strresult);
							pUnicodeStringResult->dispose();
							return flagResult;
						}
					}

					break;				// PropertyValueUnicodeString
				case ahm::PropertyValue::TYPE_INDEX:
					if (pConversion) {
						ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;

						return  pConversion->convertIntToString(pIndexValue->getIndex(), strresult);
					} break;
				}
			}
			return false;
		}
		bool setStringValue(ahm::PropertyValue* pValue, const std::string& strval, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_STRING:
				{
					ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*)pValue;
					pStringValue->setValue(strval.c_str());
					return true;
				} break;
				case ahm::PropertyValue::TYPE_INT:
					if (pConversion) {
						ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
						iop::int32 ival = 0;
						if (pConversion->convertStringToInt(strval.c_str(), ival)) {
							pIntValue->setValue(ival);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_FLOAT:
					if (pConversion) {
						ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;
						iop::float64 f64val = 0;
						if (pConversion->convertStringToFloat(strval.c_str(), f64val)) {
							pFloatValue->setValue(f64val);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_BOOL:
					if (pConversion) {
						ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
						bool flag = false;
						if (pConversion->convertStringToBool(strval.c_str(), flag)) {
							pBoolValue->setValue(flag);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_UNICODE_STRING:
					if (pConversion) {
						ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*) pValue;
						std::wstring wstr;
						if (pConversion->convertStringToUnicodeString(strval.c_str(), wstr)) {
							pUnicodeStringValue->setValue(wstr.c_str());
							return true;
						}
					}
					break;
				case ahm::PropertyValue::TYPE_INDEX:
					if (pConversion) {
						ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;

						iop::int32 ival = 0;
						if (pConversion->convertStringToInt(strval.c_str(), ival)) {
							pIndexValue->setIndex(ival);
							return true;
						}
					}
					break;
				}
			}
			return false;
		}


		bool getStringValue(ahm::Property* pProperty, std::string& strresult, IValueConversion* pConversion /* = 0 */) {
			return pProperty != 0 ? getStringValue(pProperty->value(), strresult, pConversion) : false;
		}
		bool setStringValue(ahm::Property* pProperty, const std::string& strvalue, IValueConversion* pConversion /* = 0 */) {
			return pProperty != 0 ? setStringValue(pProperty->value(), strvalue, pConversion) : false;
		}
		bool getStringValue(ahm::Properties* pProperties, iop::int32 id, std::string& strresult, IValueConversion* pConversion /* = 0 */) {
			return getStringValue(findProperty(pProperties, id), strresult, pConversion);
		}

		bool setStringValue(ahm::Properties* pProperties, iop::int32 id, const std::string& strvalue, IValueConversion* pConversion /* = 0 */) {
			return setStringValue(findProperty(pProperties, id), strvalue, pConversion);
		}


		bool getUnicodeStringValue(ahm::PropertyValue* pValue, std::wstring& wstrresult, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_UNICODE_STRING:
				{
					ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*)pValue;
					ahm::UnicodeStringResult* pStrResult = pUnicodeStringValue->getValue();
					if (pStrResult) {
						wstrresult = pStrResult->value();
						pStrResult->dispose();
						return true;
					}
				} break;
				case ahm::PropertyValue::TYPE_INT:
					if (pConversion) {
						ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
						return pConversion->convertIntToUnicodeString(pIntValue->getValue(), wstrresult);
					} break;
				case ahm::PropertyValue::TYPE_FLOAT:
					if (pConversion) {
						ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;
						return pConversion->convertFloatToUnicodeString(pFloatValue->getValue(), wstrresult);
					} break;
				case ahm::PropertyValue::TYPE_BOOL:
					if (pConversion) {
						ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
						return pConversion->convertBoolToUnicodeString(pBoolValue->getValue(), wstrresult);

					} break;
				case ahm::PropertyValue::TYPE_STRING:
					if (pConversion) {
						ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*) pValue;
						ahm::StringResult* pStringResult = pStringValue->getValue();

						if (pStringResult) {
							bool flagResult = pConversion->convertStringToUnicodeString(pStringResult->value(), wstrresult);
							pStringResult->dispose();
							return flagResult;
						}
					}

					break;				// PropertyValueUnicodeString
				case ahm::PropertyValue::TYPE_INDEX:
					if (pConversion) {
						ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;
						return  pConversion->convertIntToUnicodeString(pIndexValue->getIndex(), wstrresult);
					} break;
				}
			}
			return false;
		}

		bool setUnicodeStringValue(ahm::PropertyValue* pValue, const std::wstring& wstrval, IValueConversion* pConversion /* = 0 */) {
			if (pValue) {
				switch (pValue->derivedType()) {
				case ahm::PropertyValue::TYPE_UNICODE_STRING:
				{
					ahm::PropertyValueUnicodeString* pUnicodeStringValue = (ahm::PropertyValueUnicodeString*)pValue;
					pUnicodeStringValue->setValue(wstrval.c_str());
					return true;
				} break;
				case ahm::PropertyValue::TYPE_INT:
					if (pConversion) {
						ahm::PropertyValueInt* pIntValue = (ahm::PropertyValueInt*)pValue;
						iop::int32 ival = 0;
						if (pConversion->convertUnicodeStringToInt(wstrval.c_str(), ival)) {
							pIntValue->setValue(ival);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_FLOAT:
					if (pConversion) {
						ahm::PropertyValueFloat* pFloatValue = (ahm::PropertyValueFloat*)pValue;
						iop::float64 f64val = 0;
						if (pConversion->convertUnicodeStringToFloat(wstrval.c_str(), f64val)) {
							pFloatValue->setValue(f64val);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_BOOL:
					if (pConversion) {
						ahm::PropertyValueBool* pBoolValue = (ahm::PropertyValueBool*)pValue;
						bool flag = false;
						if (pConversion->convertUnicodeStringToBool(wstrval.c_str(), flag)) {
							pBoolValue->setValue(flag);
							return true;
						}
					} break;
				case ahm::PropertyValue::TYPE_STRING:
					if (pConversion) {
						ahm::PropertyValueString* pStringValue = (ahm::PropertyValueString*) pValue;
						std::string str;
						if (pConversion->convertUnicodeStringToString(wstrval.c_str(), str)) {
							pStringValue->setValue(str.c_str());
							return true;
						}
					}
					break;
				case ahm::PropertyValue::TYPE_INDEX:
					if (pConversion) {
						ahm::PropertyValueIndex* pIndexValue = (ahm::PropertyValueIndex*)pValue;

						iop::int32 ival = 0;
						if (pConversion->convertUnicodeStringToInt(wstrval.c_str(), ival)) {
							pIndexValue->setIndex(ival);
							return true;
						}
					}
					break;
				}
			}
			return false;
		}

		bool getUnicodeStringValue(ahm::Property* pProperty, std::wstring& wstrresult, IValueConversion* pConversion /* = 0 */) {
			return pProperty != 0 ? getUnicodeStringValue(pProperty->value(), wstrresult, pConversion) : false;
		}
		bool setUnicodeStringValue(ahm::Property* pProperty, const std::wstring& wstrvalue, IValueConversion* pConversion /* = 0 */) {
			return pProperty != 0 ? setUnicodeStringValue(pProperty->value(), wstrvalue, pConversion) : false;
		}
		bool getUnicodeStringValue(ahm::Properties* pProperties, iop::int32 id, std::wstring& wstrresult, IValueConversion* pConversion /* = 0 */) {
			return getUnicodeStringValue(findProperty(pProperties, id), wstrresult, pConversion);
		}
		bool setUnicodeStringValue(ahm::Properties* pProperties, iop::int32 id, const std::wstring& wstrvalue, IValueConversion* pConversion /* = 0 */) {
			return setUnicodeStringValue(findProperty(pProperties, id), wstrvalue, pConversion);
		}


		bool isArray(ahm::PropertyValue* pValue) {
			if (pValue) {
				return pValue->derivedType() == ahm::PropertyValue::TYPE_FIXED_ARRAY;
			}
			return false;
		}

		bool isProperties(ahm::PropertyValue* pValue) {
			if (pValue) {
				return pValue->derivedType() == ahm::PropertyValue::TYPE_PROPERTIES;
			}
			return false;
		}

		bool isScalar(ahm::PropertyValue* pValue) {
			if (pValue) {
				return pValue->derivedType() >= ahm::PropertyValue::__TYPE_SCALAR_START
					&& pValue->derivedType() < ahm::PropertyValue::__TYPE_SCALAR_MAX;
			}
			return false;
		}

		bool isRect(ahm::PropertyValue* pValue) {
			if (pValue) {
				return pValue->derivedType() == ahm::PropertyValue::TYPE_RECT
#ifdef AHM_2_0
					|| pValue->derivedType() == ahm::PropertyValue::TYPE_INT_RECT
#endif
					|| pValue->derivedType() == ahm::PropertyValue::TYPE_FLOAT_RECT;
			}
			return false;
		}

		bool isArray(ahm::Property* pProperty) {
			return pProperty ? isArray(pProperty->value()) : false;
		}

		bool isProperties(ahm::Property* pProperty) {
			return pProperty ? isProperties(pProperty->value()) : false;
		}

		bool isScalar(ahm::Property* pProperty) {
			return pProperty ? isScalar(pProperty->value()) : false;
		}

		bool isRect(ahm::Property* pProperty) {
			return pProperty ? isRect(pProperty->value()) : false;
		}

		// array access
		ahm::PropertyValue* getIndexedValue(ahm::PropertyValue* pValue, iop::int32 index) {
			if (isArray(pValue)) {
				ahm::PropertyValueArray* pArray = (ahm::PropertyValueArray*) pValue;
				if (index >= pArray->minIndex() && index <= pArray->maxIndex()) {
					return pArray->getValue(index);
				}
			}
			return 0;
		}

		ahm::PropertyValue* getIndexedValue(ahm::Property* pProperty, iop::int32 index) {
			return pProperty ? getIndexedValue(pProperty->value(), index) : 0;
		}

		bool isReadable(ahm::Property* pProperty)
		{
			return pProperty != nullptr ? isReadable(pProperty->getPropertyInfo()) : false;
		}

		bool isWritable(ahm::Property* pProperty)
		{
			return pProperty != nullptr ? isWritable(pProperty->getPropertyInfo()) : false;
		}

		ahm::PropertyInfoEnum* infoEnum(ahm::Property* pProperty) {
			return pProperty != 0 ? infoEnum(pProperty->getPropertyInfo()) : 0;
		}

		ahm::PropertyValue* getEnumValue(ahm::Property* pProperty, iop::int32 index) {
			if (infoEnum(pProperty) != 0) {
				return infoEnum(pProperty)->getOption(index);
			}
			return 0;
		}

		ahm::PropertyInfoRange* infoRange(ahm::Property* pProperty) {
			return pProperty != 0 ? infoRange(pProperty->getPropertyInfo()) : 0;
		}

		ahm::PropertyValue* minRangeValue(ahm::Property* pProperty) {
			if (infoRange(pProperty)) {
				return infoRange(pProperty)->minValue();
			}
			return 0;
		}

		ahm::PropertyValue* maxRangeValue(ahm::Property* pProperty) {
			if (infoRange(pProperty)) {
				return infoRange(pProperty)->maxValue();
			}
			return 0;
		}

		ahm::PropertyInfoSteppedRange* infoSteppedRange(ahm::Property* pProperty) {
			return pProperty != 0 ? infoSteppedRange(pProperty->getPropertyInfo()) : 0;
		}

		ahm::PropertyValue* rangeStepSizeValue(ahm::Property* pProperty) {
			if (infoSteppedRange(pProperty)) {
				return infoSteppedRange(pProperty)->stepSize();
			}
			return 0;
		}
#pragma region tools on PropertyInfo

		bool isReadable(ahm::PropertyInfo* pPropertyInfo)
		{
			if (pPropertyInfo != 0) {
				return (pPropertyInfo->accessRights() & ahm::PropertyInfo::READ) != 0;
			}
			return false;
		}

		bool isWritable(ahm::PropertyInfo* pPropertyInfo)
		{
			if (pPropertyInfo != 0) {
				return (pPropertyInfo->accessRights() & ahm::PropertyInfo::WRITE) != 0;
			}
			return false;
		}

		ahm::PropertyInfoEnum* infoEnum(ahm::PropertyInfo* pPropertyInfo) {
			if (pPropertyInfo != 0 && pPropertyInfo->derivedType() == ahm::PropertyInfo::TYPE_ENUMERATION) {
				return (ahm::PropertyInfoEnum*)pPropertyInfo;
			}
			return 0;
		}

		ahm::PropertyValue* getEnumValue(ahm::PropertyInfo* pPropertyInfo, iop::int32 index) {
			if (infoEnum(pPropertyInfo) != 0) {
				return infoEnum(pPropertyInfo)->getOption(index);
			}
			return 0;
		}

		ahm::PropertyInfoRange* infoRange(ahm::PropertyInfo* pPropertyInfo) {
			if (pPropertyInfo != 0 && (pPropertyInfo->derivedType() == ahm::PropertyInfo::TYPE_RANGE || pPropertyInfo->derivedType() == ahm::PropertyInfo::TYPE_STEPPED_RANGE)) {
				return (ahm::PropertyInfoRange*)pPropertyInfo;
			}
			return 0;
		}

		ahm::PropertyValue* minRangeValue(ahm::PropertyInfo* pPropertyInfo) {
			if (infoRange(pPropertyInfo)) {
				return infoRange(pPropertyInfo)->minValue();
			}
			return 0;
		}

		ahm::PropertyValue* maxRangeValue(ahm::PropertyInfo* pPropertyInfo) {
			if (infoRange(pPropertyInfo)) {
				return infoRange(pPropertyInfo)->maxValue();
			}
			return 0;
		}

		ahm::PropertyInfoSteppedRange* infoSteppedRange(ahm::PropertyInfo* pPropertyInfo) {
			if (pPropertyInfo != 0 && pPropertyInfo->derivedType() == ahm::PropertyInfo::TYPE_STEPPED_RANGE) {
				return (ahm::PropertyInfoSteppedRange*)pPropertyInfo;
			}
			return 0;
		}

		ahm::PropertyValue* rangeStepSizeValue(ahm::PropertyInfo* pPropertyInfo) {
			if (infoSteppedRange(pPropertyInfo)) {
				return infoSteppedRange(pPropertyInfo)->stepSize();
			}
			return 0;
		}
#pragma endregion

#if _MSC_VER >= 1400
#pragma message("Including secure versions of CRT functions available starting from VC++ version 8.0")
#endif


#ifdef USE_ANSI_C
#define _itoa_s itoa
#define _gcvt_s gcvt
#endif


		iop::int32 toInt(iop::string szval, iop::int32 dfltval/*=0*/) {
			iop::int32 result = dfltval;
			if (szval) {
				if (*szval == '#' || (*szval == '0' && toupper(*(szval + 1)) == 'X')) {
					if (*szval != '#') szval++;
					szval++;

					result = dfltval;
					{
						iop::int32 ival = 0;
						while (*szval) {
							char ch = (char)toupper(*szval);
							iop::int32 nibble = 0;
							if (ch >= '0' && ch <= '9') {
								nibble = ch - '0';
							}
							else if (ch >= 'A' && ch <= 'F') {
								nibble = ch - 'A' + 10;
							}
							else return dfltval;


							ival = (ival << 4) | (nibble & 0xf);
							szval++;

						}
						result = ival;//&0x7fffffff;

					}

				}
				else if (*szval) {
					result = atoi(szval);
				}
			}
			return result;
		}
		iop::int32 toInt(iop::unicode_string wszval, iop::int32 dfltval/*=0*/) {
			iop::int32 result = dfltval;
			if (wszval) {
				if (*wszval == L'#' || (*wszval == L'0' && toupper(*(wszval + 1)) == L'X')) {
					if (*wszval != L'#') wszval++;
					wszval++;

					result = dfltval;
					{
						iop::int32 ival = 0;
						while (*wszval) {
							wchar_t ch = (wchar_t)toupper(*wszval);
							iop::int32 nibble = 0;
							if (ch >= L'0' && ch <= L'9') {
								nibble = ch - L'0';
							}
							else if (ch >= L'A' && ch <= L'F') {
								nibble = ch - L'A' + 10;
							}
							else return dfltval;


							ival = (ival << 4) | (nibble & 0xf);
							wszval++;

						}
						result = ival;//&0x7fffffff;

					}

				}
				else if (*wszval) {
					result = _wtoi(wszval);
				}
			}
			return result;
		}


		class StandardConversion : public IValueConversion {
		public:
			static StandardConversion _instance;
			static void _test();
			static bool _convertStringToInt(iop::string szvalue, iop::int32& iresult) {
				if (szvalue) {
					iresult = toInt(szvalue, 0);
					return true;
				}
				return false;
			}

			static bool _convertIntToString(iop::int32 ival, std::string& strresult) {
				char szbuf[128];
				_itoa_s(ival, szbuf, 10);
				strresult = szbuf;
				return true;
			}

			static bool _convertStringToBool(iop::string szvalue, bool& bresult) {
				if (szvalue) {
					if (toupper(*szvalue) == 'T') {
						bresult = true;
					}
					else {
						bresult = atoi(szvalue) != 0;
					}
					return true;
				}
				return false;
			}
			static bool _convertBoolToString(bool bval, std::string& strresult) {
				strresult = bval ? "1" : "0";
				return true;
			}

			static bool _convertStringToFloat(iop::string szvalue, iop::float64& f64result) {
				if (szvalue) {
					f64result = atof(szvalue);
					return true;
				}
				return false;
			}

			static bool _convertFloatToString(iop::float64 f64val, std::string& strresult) {
				char szbuf[128];
				_gcvt_s(szbuf, f64val, 32);
				strresult = szbuf;
				return true;
			}

			static bool _convertUnicodeStringToInt(iop::unicode_string wszvalue, iop::int32& iresult) {
				if (wszvalue) {
					iresult = toInt(wszvalue, 0);
					return true;
				}
				return false;
			}

			static bool _convertIntToUnicodeString(iop::int32 ival, std::wstring& wstrresult) {
				wchar_t wszbuf[128];
				_itow_s(ival, wszbuf, 10);
				wstrresult = wszbuf;
				return true;
			}


			static bool _convertUnicodeStringToBool(iop::unicode_string wszvalue, bool& bresult) {
				if (wszvalue) {
					if (toupper(*wszvalue) == L'T') {
						bresult = true;
					}
					else {
						bresult = _wtoi(wszvalue) != 0;
					}
					return true;
				}
				return false;
			}
			static bool _convertBoolToUnicodeString(bool bval, std::wstring& wstrresult) {
				wstrresult = bval ? L"1" : L"0";
				return true;
			}

			static bool _convertUnicodeStringToFloat(iop::unicode_string wszvalue, iop::float64& f64result) {
				if (wszvalue) {
					f64result = _wtof(wszvalue);
					return true;
				}
				return false;
			}

			static bool _convertFloatToUnicodeString(iop::float64 f64val, std::wstring& wstrresult) {
				std::string strtemp;
				_convertFloatToString(f64val, strtemp);
				return _convertStringToUnicodeString(strtemp.c_str(), wstrresult);
			}

			static bool _convertStringToUnicodeString(iop::string szvalue, std::wstring& wstrresult) {
				if (szvalue) {
					size_t len = strlen(szvalue);
					wchar_t* pwszbuffer = new wchar_t[len + 1];
					if (pwszbuffer != 0) {
#ifdef USE_ANSI_C
						mstowcs(pwszbuffer, szvalue, len);
#else
						size_t lenout = 0;
						mbstowcs_s(&lenout, pwszbuffer, len + 1, szvalue, len);
#endif
						pwszbuffer[len] = L'\0';
						wstrresult = pwszbuffer;
						delete[] pwszbuffer;
						return true;
					}
				}
				return false;
			}

			static bool _convertUnicodeStringToString(iop::unicode_string wszvalue, std::string& strresult) {
				if (wszvalue) {
					size_t len = wcslen(wszvalue);

					char* pszbuffer = new char[len + 1];
					if (pszbuffer != 0) {
#ifdef USE_ANSI_C
						wcstombs(pszbuffer, wszvalue, len);
#else
						size_t lenout = 0;
						wcstombs_s(&lenout, pszbuffer, len + 1, wszvalue, len);
#endif
						pszbuffer[len] = '\0';
						strresult = pszbuffer;
						delete[] pszbuffer;
						return true;
					}
				}
				return false;
			}

			static bool _convertIntToBool(iop::int32 ival, bool& bresult) {
				bresult = ival != 0;
				return true;
			}
			static bool _convertBoolToInt(bool bval, iop::int32& iresult) {
				iresult = bval ? 1 : 0;
				return true;
			}

			static bool _convertFloatToInt(iop::float64 f64val, iop::int32& iresult) {
				// standard  C double int conversion
				iresult = (iop::int32) f64val;
				return true;
			}
			static bool _convertIntToFloat(iop::int32 ival, iop::float64& f64result) {
				f64result = ival;
				return true;
			}

			static bool _convertFloatToBool(iop::float64 f64val, bool& bresult) {
				return _convertIntToBool((iop::int32) f64val, bresult);
			}
			static bool _convertBoolToFloat(bool bval, iop::float64& f64result) {
				f64result = bval ? 1.0 : 0.0;
				return true;
			}

			// IValueConversion

			virtual bool convertStringToInt(iop::string szvalue, iop::int32& iresult) {
				return _convertStringToInt(szvalue, iresult);
			}

			virtual bool convertIntToString(iop::int32 ival, std::string& strresult) {
				return _convertIntToString(ival, strresult);
			}

			virtual bool convertStringToBool(iop::string szvalue, bool& bresult) {
				return _convertStringToBool(szvalue, bresult);
			}
			virtual bool convertBoolToString(bool bval, std::string& strresult) {
				return _convertBoolToString(bval, strresult);
			}

			virtual bool convertStringToFloat(iop::string szvalue, iop::float64& f64result) {
				return _convertStringToFloat(szvalue, f64result);
			}

			virtual bool convertFloatToString(iop::float64 f64val, std::string& strresult) {
				return _convertFloatToString(f64val, strresult);
			}

			virtual bool convertUnicodeStringToInt(iop::unicode_string wszvalue, iop::int32& iresult) {
				return _convertUnicodeStringToInt(wszvalue, iresult);
			}

			virtual bool convertIntToUnicodeString(iop::int32 ival, std::wstring& wstrresult) {
				return _convertIntToUnicodeString(ival, wstrresult);
			}

			virtual bool convertUnicodeStringToBool(iop::unicode_string wszvalue, bool& bresult) {
				return _convertUnicodeStringToBool(wszvalue, bresult);
			}

			virtual bool convertBoolToUnicodeString(bool bval, std::wstring& wstrresult) {
				return _convertBoolToUnicodeString(bval, wstrresult);
			}

			virtual bool convertUnicodeStringToFloat(iop::unicode_string wszvalue, iop::float64& f64result) {
				return _convertUnicodeStringToFloat(wszvalue, f64result);
			}

			virtual bool convertFloatToUnicodeString(iop::float64 f64val, std::wstring& wstrresult) {
				return _convertFloatToUnicodeString(f64val, wstrresult);
			}

			virtual bool convertStringToUnicodeString(iop::string szvalue, std::wstring& wstrresult) {
				return _convertStringToUnicodeString(szvalue, wstrresult);
			}

			virtual bool convertUnicodeStringToString(iop::unicode_string wszvalue, std::string& strresult) {
				return _convertUnicodeStringToString(wszvalue, strresult);
			}

			virtual bool convertIntToBool(iop::int32 ival, bool& bresult) {
				return _convertIntToBool(ival, bresult);
			}

			virtual bool convertBoolToInt(bool bval, iop::int32& iresult) {
				return _convertBoolToInt(bval, iresult);
			}

			virtual bool convertFloatToInt(iop::float64 f64val, iop::int32& iresult) {
				return _convertFloatToInt(f64val, iresult);
			}
			virtual bool convertIntToFloat(iop::int32 ival, iop::float64& f64result) {
				return _convertIntToFloat(ival, f64result);
			}

			virtual bool convertFloatToBool(iop::float64 f64val, bool& bresult) {
				return _convertFloatToBool(f64val, bresult);
			}
			virtual bool convertBoolToFloat(bool bval, iop::float64& f64result) {
				return _convertBoolToFloat(bval, f64result);
			}
		};



		StandardConversion StandardConversion::_instance;  // static instance



		// todo - add conversion option here, too
		void copy_scalar(ahm::PropertyValue* pTarget, ahm::PropertyValue* pSource) {

			if (pSource && pTarget) {

				if (pSource->derivedType() == pTarget->derivedType()) {

					switch (pSource->derivedType()) {
					case ahm::PropertyValue::TYPE_STRING:
					{
						ahm::PropertyValueString* pStringValueSource = (ahm::PropertyValueString*) pSource;
						ahm::PropertyValueString* pStringValueTarget = (ahm::PropertyValueString*) pTarget;
						ahm::StringResult* pStrResult = pStringValueSource->getValue();
						if (pStrResult) {
							pStringValueTarget->setValue(pStrResult->value());
							pStrResult->dispose();
						}
					} break;
					case ahm::PropertyValue::TYPE_INT:
					{
						ahm::PropertyValueInt* pIntValueSource = (ahm::PropertyValueInt*)pSource;
						ahm::PropertyValueInt* pIntValueTarget = (ahm::PropertyValueInt*)pTarget;
						pIntValueTarget->setValue(pIntValueSource->getValue());
					} break;
					case ahm::PropertyValue::TYPE_FLOAT:
					{
						ahm::PropertyValueFloat* pFloatValueSource = (ahm::PropertyValueFloat*)pSource;
						ahm::PropertyValueFloat* pFloatValueTarget = (ahm::PropertyValueFloat*)pTarget;
						pFloatValueTarget->setValue(pFloatValueSource->getValue());
					} break;
					case ahm::PropertyValue::TYPE_BOOL:
					{
						ahm::PropertyValueBool* pBoolValueSource = (ahm::PropertyValueBool*)pSource;
						ahm::PropertyValueBool* pBoolValueTarget = (ahm::PropertyValueBool*)pTarget;
						pBoolValueTarget->setValue(pBoolValueSource->getValue());
					} break;
					case ahm::PropertyValue::TYPE_UNICODE_STRING:
					{
						auto pStringValueSource = (ahm::PropertyValueUnicodeString*) pSource;
						auto pStringValueTarget = (ahm::PropertyValueUnicodeString*) pTarget;
						auto pStrResult = pStringValueSource->getValue();
						if (pStrResult)
						{
							pStringValueTarget->setValue(pStrResult->value());
							pStrResult->dispose();
						}
					} break;

					case ahm::PropertyValue::TYPE_INDEX:
					{
						ahm::PropertyValueIndex* pIndexValueSource = (ahm::PropertyValueIndex*)pSource;
						ahm::PropertyValueIndex* pIndexValueTarget = (ahm::PropertyValueIndex*)pTarget;
						pIndexValueTarget->setIndex(pIndexValueSource->getIndex());
					} break;

					}
				}

				else {
					throwException(ahm::ERROR_CLASS_GENERAL_ERROR, ahm::ERROR_CODE_PARAMETER_INVALID,
						"copy_scalar mismatching property types");
				}
			}
		}


		bool getFloat(iop::float64& fltTarget, ahm::PropertyValue* pSource) {
			if (isScalar(pSource)) {
				switch (pSource->derivedType()) {
				case ahm::PropertyValue::TYPE_INT:
				{
					ahm::PropertyValueInt* pIntValueSource = (ahm::PropertyValueInt*)pSource;

					fltTarget = pIntValueSource->getValue();
					return true;
				}
				case ahm::PropertyValue::TYPE_FLOAT:
				{
					ahm::PropertyValueFloat* pFloatValueSource = (ahm::PropertyValueFloat*)pSource;
					fltTarget = pFloatValueSource->getValue();
					return true;
				}

				}
			}
			return false;
		}

		bool setFloat(ahm::PropertyValue* pTarget, iop::float64 fltval) {
			if (isScalar(pTarget)) {
				switch (pTarget->derivedType()) {
				case ahm::PropertyValue::TYPE_INT:
				{
					ahm::PropertyValueInt* pIntValueTarget = (ahm::PropertyValueInt*)pTarget;
					pIntValueTarget->setValue((iop::int32) fltval);
					return true;
				}
				case ahm::PropertyValue::TYPE_FLOAT:
				{
					ahm::PropertyValueFloat* pFloatValueTarget = (ahm::PropertyValueFloat*)pTarget;
					pFloatValueTarget->setValue(fltval);
					return true;
				}

				}
			}
			return false;
		}

		void setRectValues(ahm::PropertyValue* pTarget, ahm::PropertyValue* pSourceLeft, ahm::PropertyValue* pSourceTop, ahm::PropertyValue* pSourceRight, ahm::PropertyValue* pSourceBottom) {
			if (pTarget) {
				if (pTarget->derivedType() == ahm::PropertyValue::TYPE_RECT) {
					ahm::PropertyValueRect* pTargetRect = (ahm::PropertyValueRect*) pTarget;
					copy_scalar(pTargetRect->left(), pSourceLeft);
					copy_scalar(pTargetRect->top(), pSourceTop);
					copy_scalar(pTargetRect->right(), pSourceRight);
					copy_scalar(pTargetRect->bottom(), pSourceBottom);
				}
				else  if (pTarget->derivedType() == ahm::PropertyValue::TYPE_FLOAT_RECT) {
					ahm::PropertyValueFloatRect* pTargetRect = (ahm::PropertyValueFloatRect*) pTarget;
					ahm::FLOAT_RECT frect = { 0,0,0,0 };
					getFloat(frect.left, pSourceLeft);
					getFloat(frect.top, pSourceTop);
					getFloat(frect.right, pSourceRight);
					getFloat(frect.bottom, pSourceBottom);

					// change
					pTargetRect->setValue(frect);
				}
			}
		}

		void copy_rect(ahm::PropertyValue* pTarget, ahm::PropertyValue* pSource) {
			if (isRect(pTarget) && isRect(pSource)) {
				// same types!
				if (pTarget->derivedType() == ahm::PropertyValue::TYPE_RECT
					&& pSource->derivedType() == ahm::PropertyValue::TYPE_RECT)
				{
					ahm::PropertyValueRect* pSourceRect = (ahm::PropertyValueRect*) pSource;
					ahm::PropertyValueRect* pTargetRect = (ahm::PropertyValueRect*) pTarget;
					copy_scalar(pTargetRect->left(), pSourceRect->left());
					copy_scalar(pTargetRect->top(), pSourceRect->top());
					copy_scalar(pTargetRect->right(), pSourceRect->right());
					copy_scalar(pTargetRect->bottom(), pSourceRect->bottom());
				}
				else if (pTarget->derivedType() == ahm::PropertyValue::TYPE_FLOAT_RECT
					&& pSource->derivedType() == ahm::PropertyValue::TYPE_FLOAT_RECT)
				{
					ahm::PropertyValueFloatRect* pSourceFloatRect = (ahm::PropertyValueFloatRect*) pSource;
					ahm::PropertyValueFloatRect* pTargetFloatRect = (ahm::PropertyValueFloatRect*) pTarget;
					ahm::FLOAT_RECT frect = { 0,0,0,0 };
					pSourceFloatRect->getValue(frect);
					pTargetFloatRect->setValue(frect);
				}
#ifdef AHM_2_0
				else if (pTarget->derivedType() == ahm::PropertyValue::TYPE_INT_RECT
					&& pSource->derivedType() == ahm::PropertyValue::TYPE_INT_RECT)
				{
					((ahm::PropertyValueIntRect*)pTarget)->setValue(
						((ahm::PropertyValueIntRect*)pSource)->getValue());
				}
#endif
				// different types
				else if (pTarget->derivedType() == ahm::PropertyValue::TYPE_RECT
					&& pSource->derivedType() == ahm::PropertyValue::TYPE_FLOAT_RECT)
				{
					//ahm::PropertyValueFloatRect* pSourceFloatRect =  (ahm::PropertyValueFloatRect* ) pSource;
					ahm::PropertyValueRect* pTargetRect = (ahm::PropertyValueRect*) pTarget;
					ahm::FLOAT_RECT frect = { 0,0,0,0 };

					setFloat(pTargetRect->left(), frect.left);
					setFloat(pTargetRect->top(), frect.top);
					setFloat(pTargetRect->right(), frect.right);
					setFloat(pTargetRect->bottom(), frect.bottom);

				}

				else if (pTarget->derivedType() == ahm::PropertyValue::TYPE_FLOAT_RECT
					&& pSource->derivedType() == ahm::PropertyValue::TYPE_RECT)
				{
					ahm::PropertyValueRect* pSourceRect = (ahm::PropertyValueRect*) pSource;
					ahm::PropertyValueFloatRect* pTargetFloatRect = (ahm::PropertyValueFloatRect*) pTarget;
					ahm::FLOAT_RECT frect = { 0,0,0,0 };
					getFloat(frect.left, pSourceRect->left());
					getFloat(frect.top, pSourceRect->top());
					getFloat(frect.right, pSourceRect->right());
					getFloat(frect.bottom, pSourceRect->bottom());
					pTargetFloatRect->setValue(frect);
				}
				else {
					throwException(ahm::ERROR_CLASS_GENERAL_ERROR, ahm::ERROR_CODE_PARAMETER_INVALID,
						"copy_rect mismatching property types");
				}

			}
		}


#ifdef __DO_TEST__
		void assert(bool result, const std::string& text) {
			std::cout << text.c_str() << (result ? "- passed" : "- ERROR") << std::endl;
		}
#endif



		void StandardConversion::_test() {
#ifdef __DO_TEST__

			iop::float64 pi = 3.141592654;


			bool flag = false;
			std::string str;
			std::wstring wstr;
			iop::int32 ival = -1;

			assert(_convertFloatToBool(pi, flag), "_convertFloatToBool");
			assert(flag == true, "pi is true");
			assert(_convertFloatToInt(pi, ival), "_convertFloatToInt");
			assert(ival == 3, "pi -> int is 3");

			assert(_convertFloatToString(pi, str), "_convertBoolToString");
			std::cout << "pi as string is " << str.c_str() << std::endl;

			assert(_convertFloatToUnicodeString(pi, wstr), "_convertFloatToUnicodeString");
			std::cout << "pi as wide string is " << wstr.c_str() << std::endl;
			str.clear();
			assert(_convertUnicodeStringToString(wstr.c_str(), str), "_convertUnicodeStringToString");
			std::cout << "pi as wide string  converted back is " << str.c_str() << std::endl;

			ival = 0;
			assert(_convertUnicodeStringToInt(wstr.c_str(), ival), "_convertUnicodeStringToInt");

			assert(ival == 3, "L\"3.14...\" to int is 3");


			iop::float64 f64val = -1;

			assert(_convertIntToFloat(77, f64val), "_convertIntToFloat");
			assert(f64val == 77.0, "int to float (77) ");

			f64val = -1;

			assert(_convertStringToFloat(str.c_str(), f64val), "_convertStringToFloat");
			assert(f64val == pi, "string to float is pi");

			f64val = -1;
			assert(_convertUnicodeStringToFloat(wstr.c_str(), f64val), "_convertUnicodeStringToFloat");
			assert(f64val == pi, "wide string to float is pi");

			ival = -1;


			flag = false;
			assert(_convertStringToBool("t", flag), "_convertStringToBool");
			assert(flag == true, "'t' is true");

			flag = false;
			assert(_convertStringToBool("T", flag), "_convertStringToBool");
			assert(flag == true, "'T' is true");

			flag = true;
			assert(_convertStringToBool("f", flag), "_convertStringToBool");
			assert(flag == false, "'f' is false");
			flag = true;
			assert(_convertStringToBool("f", flag), "_convertStringToBool");
			assert(flag == false, "'F' is false");

			flag = false;
			assert(_convertUnicodeStringToBool(L"t", flag), "_convertUnicodeStringToBool");
			assert(flag == true, "L't' is true");

			flag = false;
			assert(_convertUnicodeStringToBool(L"T", flag), "_convertUnicodeStringToBool");
			assert(flag == true, "L'T' is true");

			flag = true;
			assert(_convertUnicodeStringToBool(L"f", flag), "_convertUnicodeStringToBool");
			assert(flag == false, "L'f' is false");

			flag = true;
			assert(_convertUnicodeStringToBool(L"f", flag), "_convertUnicodeStringToBool");
			assert(flag == false, "L'F' is false");




			str.clear(); wstr.clear();

			ival = 0;
			assert(_convertStringToInt("0xff", ival), "_convertStringToInt");
			assert(ival == 255, "\"0xff\" is 255");
			ival = 0;
			assert(_convertStringToInt("#ff", ival), "_convertStringToInt");
			assert(ival == 255, "\"#ff\" is 255");

			ival = 0;
			assert(_convertStringToInt("255", ival), "_convertStringToInt");
			assert(ival == 255, "\255\" is 255");


			ival = 0;
			assert(_convertUnicodeStringToInt(L"0xff", ival), "_convertUnicodeStringToInt");
			assert(ival == 255, "L\"0xff\" is 255");
			ival = 0;
			assert(_convertUnicodeStringToInt(L"#ff", ival), "_convertUnicodeStringToInt");
			assert(ival == 255, "L\"#ff\" is 255");

			ival = 0;
			assert(_convertUnicodeStringToInt(L"255", ival), "_convertUnicodeStringToInt");
			assert(ival == 255, "L\"255\" is 255");
#endif
		}



		IValueConversion& _standardConversion = StandardConversion::_instance;

		void __test() {
			StandardConversion::_test();
		}

	}// end namespace

}// end namespace

#endif