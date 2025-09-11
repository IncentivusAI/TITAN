(function titanUniversalModule(root, factory) {
	if (typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if (typeof define === 'function' && define.amd)
		define([], factory);
	else if (typeof exports === 'object')
		exports["titanSlider"] = factory();
	else
		root["titanSlider"] = factory();
})(typeof self !== 'undefined' ? self : this, function() {
return /******/ (function(modules) { // titanBootstrap
/******/ 	// Cache for modules
/******/ 	var loadedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __titan_require__(moduleId) {
/******/
/******/ 		// If module is cached, return
/******/ 		if (loadedModules[moduleId]) {
/******/ 			return loadedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create new module and add to cache
/******/ 		var module = loadedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Run module
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __titan_require__);
/******/
/******/ 		// Mark as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return exports
/******/ 		return module.exports;
/******/ 	}
/******/
/******/ 	// expose module definitions
/******/ 	__titan_require__.m = modules;
/******/
/******/ 	// expose module cache
/******/ 	__titan_require__.c = loadedModules;
/******/
/******/ 	// define getter for harmony exports
/******/ 	__titan_require__.d = function(exports, name, getter) {
/******/ 		if(!__titan_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};
/******/
/******/ 	// get default export for non-harmony modules
/******/ 	__titan_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__titan_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// hasOwnProperty shortcut
/******/ 	__titan_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// public path
/******/ 	__titan_require__.p = "";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __titan_require__(__titan_require__.s = 0);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, __titan_exports__, __titan_require__) {

"use strict";
Object.defineProperty(__titan_exports__, "__esModule", { value: true });
/* harmony export (binding) */ __titan_require__.d(__titan_exports__, "isString", function() { return isString; });
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__events__ = __titan_require__(1);
var _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; };

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _typeof = typeof Symbol === "function" && typeof Symbol.iterator === "symbol" ? function (obj) { return typeof obj; } : function (obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; };

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object_
