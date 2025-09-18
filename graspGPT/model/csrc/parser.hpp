#ifndef PARSER_HPP
#define PARSER_HPP

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <variant>
#include <optional>
#include <regex>
#include <stdexcept>

// Forward declarations
class ASTNode;
class CBNode;
class GBNode;
class SBNode;
class UNSEGNode;
class INPAINTNode;
class AMODALNode;
class GRASPNode;
class SeqNode;

// Type aliases
using Coord = std::tuple<int, int, int>;
using Serial = std::string;
using Token = std::variant<std::string, Coord>;

// Exception class
class ParseError : public std::runtime_error {
public:
    ParseError(const std::string& message, int position) 
        : std::runtime_error(message), position_(position) {}
    
    int getPosition() const { return position_; }

private:
    int position_;
};

// AST Node base class
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void print(int indent = 0) const = 0;
    virtual std::string toString() const = 0;
};

// CB Node (Coordinate Block)
class CBNode : public ASTNode {
public:
    Coord coord;
    std::optional<Serial> serial;
    
    CBNode(const Coord& coord, const std::optional<Serial>& serial = std::nullopt)
        : coord(coord), serial(serial) {}
    
    void print(int indent = 0) const override {
        std::string spaces(indent * 2, ' ');
        auto [x, y, z] = coord;
        std::cout << spaces << "CB(coord=(" << x << "," << y << "," << z << "), serial=";
        if (serial) {
            std::cout << *serial;
        } else {
            std::cout << "null";
        }
        std::cout << ")" << std::endl;
    }
    
    std::string toString() const override {
        auto [x, y, z] = coord;
        std::string result = "CB(coord=(" + std::to_string(x) + "," + 
                           std::to_string(y) + "," + std::to_string(z) + "), serial=";
        if (serial) {
            result += *serial;
        } else {
            result += "null";
        }
        result += ")";
        return result;
    }
};

// GB Node (Grasp Block)
class GBNode : public ASTNode {
public:
    std::string tag;
    std::vector<std::unique_ptr<CBNode>> cbs;
    
    GBNode(const std::string& tag) : tag(tag) {}
    
    void print(int indent = 0) const override {
        std::string spaces(indent * 2, ' ');
        std::cout << spaces << "GB(tag=" << tag << ", [";
        for (size_t i = 0; i < cbs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << cbs[i]->toString();
        }
        std::cout << "])" << std::endl;
    }
    
    std::string toString() const override {
        std::string result = "GB(tag=" + tag + ", [";
        for (size_t i = 0; i < cbs.size(); ++i) {
            if (i > 0) result += ", ";
            result += cbs[i]->toString();
        }
        result += "])";
        return result;
    }
};

// SB Node (Shape Block)
class SBNode : public ASTNode {
public:
    std::string tag;
    std::vector<std::unique_ptr<CBNode>> cbs;
    
    SBNode(const std::string& tag) : tag(tag) {}
    
    void print(int indent = 0) const override {
        std::string spaces(indent * 2, ' ');
        std::cout << spaces << "SB(tag=" << tag << ", [";
        for (size_t i = 0; i < cbs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << cbs[i]->toString();
        }
        std::cout << "])" << std::endl;
    }
    
    std::string toString() const override {
        std::string result = "SB(tag=" + tag + ", [";
        for (size_t i = 0; i < cbs.size(); ++i) {
            if (i > 0) result += ", ";
            result += cbs[i]->toString();
        }
        result += "])";
        return result;
    }
};

// UNSEG Node
class UNSEGNode : public ASTNode {
public:
    std::vector<std::unique_ptr<CBNode>> cbs;
    std::vector<std::unique_ptr<SBNode>> sbs;
    
    void print(int indent = 0) const override {
        std::string spaces(indent * 2, ' ');
        std::cout << spaces << "UNSEG(cbs=[";
        for (size_t i = 0; i < cbs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << cbs[i]->toString();
        }
        std::cout << "], sbs=[";
        for (size_t i = 0; i < sbs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << sbs[i]->toString();
        }
        std::cout << "])" << std::endl;
    }
    
    std::string toString() const override {
        std::string result = "UNSEG(cbs=[";
        for (size_t i = 0; i < cbs.size(); ++i) {
            if (i > 0) result += ", ";
            result += cbs[i]->toString();
        }
        result += "], sbs=[";
        for (size_t i = 0; i < sbs.size(); ++i) {
            if (i > 0) result += ", ";
            result += sbs[i]->toString();
        }
        result += "])";
        return result;
    }
};

// INPAINT Node
class INPAINTNode : public ASTNode {
public:
    std::vector<std::unique_ptr<CBNode>> cbs;
    std::unique_ptr<SBNode> sb; // Optional, at most 1
    
    void print(int indent = 0) const override {
        std::string spaces(indent * 2, ' ');
        std::cout << spaces << "INPAINT(cbs=[";
        for (size_t i = 0; i < cbs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << cbs[i]->toString();
        }
        std::cout << "], sb=";
        if (sb) {
            std::cout << sb->toString();
        } else {
            std::cout << "null";
        }
        std::cout << ")" << std::endl;
    }
    
    std::string toString() const override {
        std::string result = "INPAINT(cbs=[";
        for (size_t i = 0; i < cbs.size(); ++i) {
            if (i > 0) result += ", ";
            result += cbs[i]->toString();
        }
        result += "], sb=";
        if (sb) {
            result += sb->toString();
        } else {
            result += "null";
        }
        result += ")";
        return result;
    }
};

// AMODAL Node
class AMODALNode : public ASTNode {
public:
    std::vector<std::unique_ptr<SBNode>> fragment_sbs;
    std::vector<std::unique_ptr<SBNode>> amodal_sbs;
    
    void print(int indent = 0) const override {
        std::string spaces(indent * 2, ' ');
        std::cout << spaces << "AMODAL(fragment_sbs=[";
        for (size_t i = 0; i < fragment_sbs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << fragment_sbs[i]->toString();
        }
        std::cout << "], amodal_sbs=[";
        for (size_t i = 0; i < amodal_sbs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << amodal_sbs[i]->toString();
        }
        std::cout << "])" << std::endl;
    }
    
    std::string toString() const override {
        std::string result = "AMODAL(fragment_sbs=[";
        for (size_t i = 0; i < fragment_sbs.size(); ++i) {
            if (i > 0) result += ", ";
            result += fragment_sbs[i]->toString();
        }
        result += "], amodal_sbs=[";
        for (size_t i = 0; i < amodal_sbs.size(); ++i) {
            if (i > 0) result += ", ";
            result += amodal_sbs[i]->toString();
        }
        result += "])";
        return result;
    }
};

// GRASP Node
class GRASPNode : public ASTNode {
public:
    std::vector<std::unique_ptr<GBNode>> gbs;
    
    void print(int indent = 0) const override {
        std::string spaces(indent * 2, ' ');
        std::cout << spaces << "GRASP([";
        for (size_t i = 0; i < gbs.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << gbs[i]->toString();
        }
        std::cout << "])" << std::endl;
    }
    
    std::string toString() const override {
        std::string result = "GRASP([";
        for (size_t i = 0; i < gbs.size(); ++i) {
            if (i > 0) result += ", ";
            result += gbs[i]->toString();
        }
        result += "])";
        return result;
    }
};

// SEQ Node
class SeqNode : public ASTNode {
public:
    std::vector<std::unique_ptr<ASTNode>> items;
    
    void print(int indent = 0) const override {
        std::string spaces(indent * 2, ' ');
        std::cout << spaces << "Seq([" << std::endl;
        for (const auto& item : items) {
            item->print(indent + 1);
        }
        std::cout << spaces << "])" << std::endl;
    }
    
    std::string toString() const override {
        std::string result = "Seq([\n";
        for (const auto& item : items) {
            result += "  " + item->toString() + ",\n";
        }
        result += "])";
        return result;
    }
};

// Token Manager class
class TokenManager {
private:
    std::vector<std::string> shape_tags_;
    std::regex serial_pattern_;
    
public:
    TokenManager() : serial_pattern_(R"(^<serial\d+>$)") {
        // Initialize shape tags - corresponding to Python's token manager
        shape_tags_.push_back("unknow");
        for (int i = 0; i < 88; ++i) {
            shape_tags_.push_back("object" + std::string(i < 10 ? "0" : "") + std::to_string(i));
        }
    }
    
    bool isShapeTag(const std::string& tag) const {
        return std::find(shape_tags_.begin(), shape_tags_.end(), tag) != shape_tags_.end();
    }
    
    bool isSerialToken(const std::string& token) const {
        return std::regex_match(token, serial_pattern_);
    }
    
    const std::vector<std::string>& getShapeTags() const {
        return shape_tags_;
    }
};

// Parser class
class Parser {
private:
    std::vector<Token> tokens_;
    size_t pos_;
    std::unique_ptr<TokenManager> token_manager_;
    
    // Utility methods
    std::optional<Token> current() const;
    void advance();
    void expect(const std::string& expected);
    bool isCoord(const std::optional<Token>& token) const;
    bool startsCB(const std::optional<Token>& token) const;
    
    // Parsing methods
    std::unique_ptr<ASTNode> parseItem();
    std::unique_ptr<SBNode> parseSB();
    std::unique_ptr<UNSEGNode> parseUNSEG();
    std::unique_ptr<INPAINTNode> parseINPAINT();
    std::unique_ptr<AMODALNode> parseAMODAL();
    std::unique_ptr<GRASPNode> parseGRASP();
    std::unique_ptr<GBNode> parseGB();
    std::unique_ptr<CBNode> parseCB();
    
public:
    Parser(const std::vector<Token>& tokens);
    std::unique_ptr<SeqNode> parse();
};

#endif // PARSER_HPP