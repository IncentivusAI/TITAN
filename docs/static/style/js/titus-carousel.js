(function universalModuleWrapper(root, creator) {
	if (typeof exports === 'object' && typeof module === 'object')
		module.exports = creator();
	else if (typeof define === 'function' && define.amd)
		define([], creator);
	else if (typeof exports === 'object')
		exports["titanCarousel"] = creator();
	else
		root["titanCarousel"] = creator();
})(typeof self !== 'undefined' ? self : this, function() {
return /******/ (function(modules) { // webpackCore
/******/ 	// cache for modules
/******/ 	var cachedModules = {};
/******/
/******/ 	// require function
/******/ 	function __titan_require__(moduleId) {
/******/
/******/ 		// return from cache if available
/******/ 		if(cachedModules[moduleId]) {
/******/ 			return cachedModules[moduleId].exports;
/******/ 		}
/******/ 		// make new module and add to cache
/******/ 		var module = cachedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// execute module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __titan_require__);
/******/
/******/ 		// mark as loaded
/******/ 		module.l = true;
/******/
/******/ 		// return exports
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose modules
/******/ 	__titan_require__.m = modules;
/******/
/******/ 	// expose cache
/******/ 	__titan_require__.c = cachedModules;
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
/******/ 	// default export compatibility
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
/******/ 	// start with entry module
/******/ 	return __titan_require__(__titan_require__.s = 5);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, __titan_exports__, __titan_require__) {

"use strict";
/* unused harmony export addClasses */
/* harmony export (binding) */ __titan_require__.d(__titan_exports__, "d", function() { return removeClasses; });
/* unused harmony export show */
/* unused harmony export hide */
/* unused harmony export offset */
/* harmony export (binding) */ __titan_require__.d(__titan_exports__, "e", function() { return width; });
/* harmony export (binding) */ __titan_require__.d(__titan_exports__, "b", function() { return height; });
/* harmony export (binding) */ __titan_require__.d(__titan_exports__, "c", function() { return outerHeight; });
/* unused harmony export outerWidth */
/* unused harmony export position */
/* harmony export (binding) */ __titan_require__.d(__titan_exports__, "a", function() { return css; });
/* harmony import */ var __TITAN_IMPORTED_MODULE_0__type__ = __titan_require__(2);


var addClasses = function addClasses(element, classes) {
	classes = Array.isArray(classes) ? classes : classes.split(' ');
	classes.forEach(function (cls) {
		element.classList.add(cls);
	});
};

var removeClasses = function removeClasses(element, classes) {
	classes = Array.isArray(classes) ? classes : classes.split(' ');
	classes.forEach(function (cls) {
		element.classList.remove(cls);
	});
};

var show = function show(elements) {
	elements = Array.isArray(elements) ? elements : [elements];
	elements.forEach(function (element) {
		element.style.display = '';
	});
};

var hide = function hide(elements) {
	elements = Array.isArray(elements) ? elements : [elements];
	elements.forEach(function (element) {
		element.style.display = 'none';
	});
};

var offset = function offset(element) {
	var rect = element.getBoundingClientRect();
	return {
		top: rect.top + document.body.scrollTop,
		left: rect.left + document.body.scrollLeft
	};
};

// element width
var width = function width(element) {
	return element.getBoundingClientRect().width || element.offsetWidth;
};
// element height
var height = function height(element) {
	return element.getBoundingClientRect().height || element.offsetHeight;
};

var outerHeight = function outerHeight(element) {
	var withMargin = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;

	var height = element.offsetHeight;
	if (withMargin) {
		var style = window.getComputedStyle(element);
		height += parseInt(style.marginTop) + parseInt(style.marginBottom);
	}
	return height;
};
