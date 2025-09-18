#include "parser.hpp"
#include <algorithm>

Parser::Parser(const std::vector<Token>& tokens) 
    : tokens_(tokens), pos_(0), token_manager_(std::make_unique<TokenManager>()) {
}

std::optional<Token> Parser::current() const {
    if (pos_ >= tokens_.size()) {
        return std::nullopt;
    }
    return tokens_[pos_];
}

void Parser::advance() {
    pos_++;
}

void Parser::expect(const std::string& expected) {
    auto tok = current();
    if (!tok) {
        throw ParseError("Expected '" + expected + "', got EOF at position " + std::to_string(pos_), pos_);
    }
    
    if (!std::holds_alternative<std::string>(*tok)) {
        throw ParseError("Expected '" + expected + "', got coordinate at position " + std::to_string(pos_), pos_);
    }
    
    const std::string& str_tok = std::get<std::string>(*tok);
    if (str_tok != expected) {
        throw ParseError("Expected '" + expected + "', got '" + str_tok + "' at position " + std::to_string(pos_), pos_);
    }
    
    advance();
}

bool Parser::isCoord(const std::optional<Token>& token) const {
    return token && std::holds_alternative<Coord>(*token);
}

bool Parser::startsCB(const std::optional<Token>& token) const {
    return isCoord(token);
}

std::unique_ptr<CBNode> Parser::parseCB() {
    auto coord_tok = current();
    if (!isCoord(coord_tok)) {
        throw ParseError("Expected coordinate tuple at position " + std::to_string(pos_), pos_);
    }
    
    Coord coord = std::get<Coord>(*coord_tok);
    advance();
    
    std::optional<Serial> serial;
    auto next_tok = current();
    if (next_tok && std::holds_alternative<std::string>(*next_tok)) {
        const std::string& str_tok = std::get<std::string>(*next_tok);
        if (token_manager_->isSerialToken(str_tok)) {
            serial = str_tok;
            advance();
        }
    }
    
    return std::make_unique<CBNode>(coord, serial);
}

std::unique_ptr<GBNode> Parser::parseGB() {
    expect("grasp");
    
    auto tag_tok = current();
    if (!tag_tok || !std::holds_alternative<std::string>(*tag_tok)) {
        throw ParseError("Expected shape tag after 'grasp' at position " + std::to_string(pos_), pos_);
    }
    
    const std::string& tag = std::get<std::string>(*tag_tok);
    if (!token_manager_->isShapeTag(tag)) {
        throw ParseError("Expected shape tag after 'grasp', got '" + tag + "' at position " + std::to_string(pos_), pos_);
    }
    
    advance();
    
    auto gb = std::make_unique<GBNode>(tag);
    
    if (!startsCB(current())) {
        throw ParseError("GB must contain at least one CB after tag at position " + std::to_string(pos_), pos_);
    }
    
    while (startsCB(current())) {
        gb->cbs.push_back(parseCB());
    }
    
    return gb;
}

std::unique_ptr<SBNode> Parser::parseSB() {
    auto tag_tok = current();
    if (!tag_tok || !std::holds_alternative<std::string>(*tag_tok)) {
        throw ParseError("Expected shape tag at position " + std::to_string(pos_), pos_);
    }
    
    const std::string& tag = std::get<std::string>(*tag_tok);
    if (!token_manager_->isShapeTag(tag)) {
        throw ParseError("Expected shape tag, got '" + tag + "' at position " + std::to_string(pos_), pos_);
    }
    
    advance();
    
    auto sb = std::make_unique<SBNode>(tag);
    
    if (!startsCB(current())) {
        throw ParseError("SB must contain at least one CB after tag at position " + std::to_string(pos_), pos_);
    }
    
    while (startsCB(current())) {
        sb->cbs.push_back(parseCB());
    }
    
    return sb;
}

std::unique_ptr<UNSEGNode> Parser::parseUNSEG() {
    expect("unlabel");
    
    auto unseg = std::make_unique<UNSEGNode>();
    
    // Parse CBs
    while (startsCB(current())) {
        unseg->cbs.push_back(parseCB());
    }
    
    expect("segment");
    
    // Parse SBs
    auto current_tok = current();
    while (current_tok && std::holds_alternative<std::string>(*current_tok)) {
        const std::string& str_tok = std::get<std::string>(*current_tok);
        if (token_manager_->isShapeTag(str_tok)) {
            unseg->sbs.push_back(parseSB());
            current_tok = current();
        } else {
            break;
        }
    }
    
    expect("endunseg");
    
    return unseg;
}

std::unique_ptr<INPAINTNode> Parser::parseINPAINT() {
    expect("fragment");
    
    auto inpaint = std::make_unique<INPAINTNode>();
    
    // Parse CBs
    while (startsCB(current())) {
        inpaint->cbs.push_back(parseCB());
    }
    
    expect("inpaint");
    
    // Parse at most 1 SB
    auto current_tok = current();
    if (current_tok && std::holds_alternative<std::string>(*current_tok)) {
        const std::string& str_tok = std::get<std::string>(*current_tok);
        if (token_manager_->isShapeTag(str_tok)) {
            inpaint->sb = parseSB();
            
            // Check for extra SBs
            current_tok = current();
            if (current_tok && std::holds_alternative<std::string>(*current_tok)) {
                const std::string& next_str = std::get<std::string>(*current_tok);
                if (token_manager_->isShapeTag(next_str)) {
                    throw ParseError("INPAINT can have at most 1 SB at position " + std::to_string(pos_), pos_);
                }
            }
        }
    }
    
    expect("endinpaint");
    
    return inpaint;
}

std::unique_ptr<AMODALNode> Parser::parseAMODAL() {
    expect("tagfragment");
    
    auto amodal = std::make_unique<AMODALNode>();
    
    // Parse fragment SBs
    auto current_tok = current();
    while (current_tok && std::holds_alternative<std::string>(*current_tok)) {
        const std::string& str_tok = std::get<std::string>(*current_tok);
        if (token_manager_->isShapeTag(str_tok)) {
            amodal->fragment_sbs.push_back(parseSB());
            current_tok = current();
        } else {
            break;
        }
    }
    
    expect("amodal");
    
    // Parse amodal SBs
    current_tok = current();
    while (current_tok && std::holds_alternative<std::string>(*current_tok)) {
        const std::string& str_tok = std::get<std::string>(*current_tok);
        if (token_manager_->isShapeTag(str_tok)) {
            amodal->amodal_sbs.push_back(parseSB());
            current_tok = current();
        } else {
            break;
        }
    }
    
    expect("endamodal");
    
    return amodal;
}

std::unique_ptr<GRASPNode> Parser::parseGRASP() {
    expect("detectgrasp");
    
    auto grasp = std::make_unique<GRASPNode>();
    
    auto current_tok = current();
    while (current_tok && std::holds_alternative<std::string>(*current_tok)) {
        const std::string& str_tok = std::get<std::string>(*current_tok);
        if (str_tok == "grasp") {
            grasp->gbs.push_back(parseGB());
            current_tok = current();
        } else {
            break;
        }
    }
    
    return grasp;
}

std::unique_ptr<ASTNode> Parser::parseItem() {
    auto tok = current();
    if (!tok || !std::holds_alternative<std::string>(*tok)) {
        throw ParseError("Expected string token at position " + std::to_string(pos_), pos_);
    }
    
    const std::string& str_tok = std::get<std::string>(*tok);
    
    if (token_manager_->isShapeTag(str_tok)) {
        return parseSB();
    }
    if (str_tok == "unlabel") {
        return parseUNSEG();
    }
    if (str_tok == "fragment") {
        return parseINPAINT();
    }
    if (str_tok == "tagfragment") {
        return parseAMODAL();
    }
    if (str_tok == "detectgrasp") {
        return parseGRASP();
    }
    
    throw ParseError("Unexpected token '" + str_tok + "' at position " + std::to_string(pos_), pos_);
}

std::unique_ptr<SeqNode> Parser::parse() {
    auto seq = std::make_unique<SeqNode>();
    
    auto current_tok = current();
    while (current_tok && !(std::holds_alternative<std::string>(*current_tok) && 
           std::get<std::string>(*current_tok) == "end")) {
        
        try {
            seq->items.push_back(parseItem());
        } catch (const ParseError& e) {
            // On error, break and return partial results (like Python version)
            std::cerr << "ParseError: " << e.what() << std::endl;
            break;
        }
        current_tok = current();
    }
    
    // Consume 'end' if present
    if (current_tok && std::holds_alternative<std::string>(*current_tok) && 
        std::get<std::string>(*current_tok) == "end") {
        advance();
    }
    
    return seq;
}